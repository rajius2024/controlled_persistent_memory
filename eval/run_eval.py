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

    def __init__(self, k: int = 15, token_cap: int = 8000, options_cap_chars: int = 2500):
        self.k = k
        self.token_cap = token_cap
        self.options_cap_chars = options_cap_chars

    def answer(self, ep: dict):
        context = (ep.get("sliced_context", "") or "")[-self.token_cap:]
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
        return pred, raw_text, [], [], 0, 0


class VanillaRAGMethod:
    """
    Vanilla RAG baseline wrapper that reuses Aravindan's retrieval pipeline:
      - embed_with_cache() from baselines.embeddings
      - retrieve_top_k() + build_prompt() from baselines.vanilla_rag
    """

    def __init__(self, k: int = 15, token_cap: int = 8000):
        self.k = k
        self.token_cap = token_cap

        from baselines.embeddings import embed_with_cache
        from baselines.vanilla_rag import retrieve_top_k, build_prompt

        self.embed_with_cache = embed_with_cache
        self.retrieve_top_k = retrieve_top_k
        self.build_prompt = build_prompt

    def answer(self, ep: dict):
        turns = ep.get("turns", [])
        if not turns:
            return "", "", [], [], 0, 0

        texts = [f"{t['role'].capitalize()}: {t['content']}" for t in turns]
        embeddings = self.embed_with_cache(texts, cache_key=f"ep_{ep.get('question_id', '')}")

        enhanced_query = f"{ep.get('question', '')} {ep.get('options', '')}"
        retrieved = self.retrieve_top_k(enhanced_query, turns, embeddings, k=self.k)

        prompt = self.build_prompt(ep, retrieved)
        raw_text = call_llm(prompt)
        pred = normalize_label(raw_text)

        retrieved_texts = [r["text"] for r in retrieved]
        retrieved_scores = [r["sim_score"] for r in retrieved]
        stored_count = len(turns)
        superseded_count = 0

        return pred, raw_text, retrieved_texts, retrieved_scores, stored_count, superseded_count


class ControlledMethod:
    """
    Placeholder controlled-method wrapper for Day 4 runner standardization.
    Replace internals with Vaarunya's controlled retrieval implementation once ready.
    """

    def __init__(self, k: int = 15, token_cap: int = 8000):
        self.k = k
        self.token_cap = token_cap

    def answer(self, ep: dict):
        turns = ep.get("turns", [])
        context = (ep.get("sliced_context", "") or "")[-self.token_cap:]
        options = (ep.get("options", "") or "")[:2500]
        question = ep.get("question", "") or ""

        prompt = (
            f"Conversation history:\n{context}\n\n"
            f"Active Memories:\n\n"
            f"Question: {question}\n"
            f"Options:\n{options}\n\n"
            f"Output format must be EXACTLY one of: (a) (b) (c) (d). No other text."
        )

        raw_text = call_llm(prompt)
        pred = normalize_label(raw_text)

        retrieved_texts = []
        retrieved_scores = []
        stored_count = len(turns)
        superseded_count = 0

        return pred, raw_text, retrieved_texts, retrieved_scores, stored_count, superseded_count


def build_output_path(method_name: str, out_dir: str) -> str:
    filename_map = {
        "no_memory": "no_memory_dev10.jsonl",
        "vanilla_rag": "vanilla_rag_dev10.jsonl",
        "controlled": "controlled_dev10.jsonl",
    }
    return os.path.join(out_dir, filename_map[method_name])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--method",
        choices=["no_memory", "vanilla_rag", "controlled"],
        required=True,
        help="Method to run",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=15,
        help="Top-k retrieval for retrieval-based methods",
    )
    parser.add_argument(
        "--token_cap",
        type=int,
        default=8000,
        help="Character cap for context used in prompt construction",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="results",
        help="Directory to write output jsonl logs",
    )
    parser.add_argument(
        "--episodes_path",
        type=str,
        default=os.path.join("data", "dev_latest.jsonl"),
        help="Path to episode dataset jsonl",
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    if args.method == "no_memory":
        method = NoMemoryMethod(k=args.k, token_cap=args.token_cap)
    elif args.method == "vanilla_rag":
        method = VanillaRAGMethod(k=args.k, token_cap=args.token_cap)
    else:
        method = ControlledMethod(k=args.k, token_cap=args.token_cap)

    out_path = build_output_path(args.method, args.out_dir)

    correct = 0
    total = 0

    with open(out_path, "w", encoding="utf-8") as out:
        for ep in iter_jsonl(args.episodes_path):
            (
                pred,
                raw_text,
                retrieved_texts,
                retrieved_scores,
                stored_count,
                superseded_count,
            ) = method.answer(ep)

            gold = normalize_label(ep.get("answer", ""))
            correct_flag = is_correct(pred, gold)
            correct += int(correct_flag)
            total += 1

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
                "k": args.k,
                "token_cap": args.token_cap,
            }

            out.write(json.dumps(record) + "\n")

    acc = correct / total if total else 0.0
    print(f"Accuracy: {correct}/{total} = {acc:.3f}")
    print(f"Wrote logs to: {out_path}")


if __name__ == "__main__":
    main()
    