# eval/run_eval.py
import argparse
import json
import os

from utils.llm import call_llm, normalize_label, is_correct


def iter_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


class NoMemoryMethod:
    """
    No-memory baseline using shared utils.llm call_llm().
    Uses sliced_context (truncated) + question + options.
    """

    def __init__(self, context_cap_chars: int = 8000, options_cap_chars: int = 2500):
        self.context_cap_chars = context_cap_chars
        self.options_cap_chars = options_cap_chars

    def answer(self, ep: dict):
        context = (ep.get("sliced_context", "") or "")[-self.context_cap_chars:]
        options = (ep.get("options", "") or "")[:self.options_cap_chars]
        question = ep.get("question", "") or ""

        prompt = (
            f"Conversation history:\n{context}\n\n"
            f"Question: {question}\n"
            f"Options:\n{options}\n\n"
            f"Output format must be EXACTLY one of: (a) (b) (c) (d). No other text."
        )

        raw_text = call_llm(prompt)
        pred = normalize_label(raw_text)
        return pred, raw_text, [], []


class VanillaRAGMethod:
    """
    Vanilla RAG baseline wrapper that reuses the shared retrieval pipeline.
    """

    def __init__(self, top_k: int = 15):
        self.top_k = top_k

        from baselines.embeddings import embed_with_cache
        from baselines.vanilla_rag import retrieve_top_k, build_prompt

        self.embed_with_cache = embed_with_cache
        self.retrieve_top_k = retrieve_top_k
        self.build_prompt = build_prompt

    def answer(self, ep: dict):
        turns = ep.get("turns", [])
        if not turns:
            return "", "", [], []

        texts = [f"{t['role'].capitalize()}: {t['content']}" for t in turns]
        embeddings = self.embed_with_cache(texts, cache_key=f"ep_{ep.get('question_id', '')}")

        enhanced_query = f"{ep.get('question', '')} {ep.get('options', '')}"
        retrieved = self.retrieve_top_k(enhanced_query, turns, embeddings, k=self.top_k)

        prompt = self.build_prompt(ep, retrieved)
        raw_text = call_llm(prompt)
        pred = normalize_label(raw_text)

        retrieved_texts = [r["text"] for r in retrieved]
        retrieved_scores = [r["sim_score"] for r in retrieved]
        return pred, raw_text, retrieved_texts, retrieved_scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["no_memory", "vanilla_rag", "controlled"], required=True)
    parser.add_argument("--top_k", type=int, default=15, help="Top-k retrieval for retrieval-based methods")
    parser.add_argument("--episodes_path", type=str, default=os.path.join("data", "dev_10.jsonl"))
    args = parser.parse_args()

    os.makedirs("results", exist_ok=True)

    if args.method == "no_memory":
        method = NoMemoryMethod()
        out_path = os.path.join("results", "no_memory_dev10.jsonl")
        top_k = 0

    elif args.method == "vanilla_rag":
        method = VanillaRAGMethod(top_k=args.top_k)
        out_path = os.path.join("results", "vanilla_rag_dev10.jsonl")
        top_k = args.top_k

    elif args.method == "controlled":
        from eval.methods.controlled import ControlledMethod

        method = ControlledMethod(
            run_root=os.path.join("results", "controlled_runs"),
            top_k=args.top_k,
        )
        out_path = os.path.join("results", "controlled_dev10.jsonl")
        top_k = args.top_k

    else:
        raise ValueError(f"Unsupported method: {args.method}")

    correct = 0
    total = 0

    with open(out_path, "w", encoding="utf-8") as out:
        for ep in iter_jsonl(args.episodes_path):
            pred, raw_text, retrieved_texts, retrieved_scores = method.answer(ep)

            gold = normalize_label(ep.get("answer", ""))
            correct_flag = is_correct(pred, gold)
            correct += int(correct_flag)
            total += 1

            stored_count = len(ep.get("turns", [])) if args.method == "vanilla_rag" else 0
            superseded_count = 0

            if args.method == "controlled" and getattr(method, "last_trace", None):
                stored_count = method.last_trace.get("stored_memories_count", 0)
                superseded_count = method.last_trace.get("superseded_count", 0)

            record = {
                "episode_id": ep.get("episode_id"),
                "question_id": ep.get("question_id"),
                "method": args.method,
                "prediction_raw": raw_text,
                "prediction": pred,
                "gold_raw": ep.get("answer", ""),
                "gold": gold,
                "correct": correct_flag,
                "retrieved": retrieved_texts,
                "retrieved_scores": retrieved_scores,
                "stored_count": stored_count,
                "retrieved_count": len(retrieved_texts),
                "superseded_count": superseded_count,
                "question_type": ep.get("question_type"),
                "topic": ep.get("topic", ""),
                "top_k": top_k,
            }

            out.write(json.dumps(record) + "\n")

    acc = correct / total if total else 0.0
    print(f"Accuracy: {correct}/{total} = {acc:.3f}")
    print(f"Wrote logs to: {out_path}")


if __name__ == "__main__":
    main()
