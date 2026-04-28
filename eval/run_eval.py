# eval/run_eval.py
import argparse
import json
import os
import time

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


class SummaryOnlyMethod:
    """
    Summary-only baseline using shared utils.llm call_llm().

    Uses a compressed summary field if available.
    If no summary field exists, falls back to sliced_context so the run does not crash.
    """

    def __init__(self, k: int = 15, token_cap: int = 8000, options_cap_chars: int = 2500):
        self.k = k
        self.token_cap = token_cap
        self.options_cap_chars = options_cap_chars

    def answer(self, ep: dict):
        summary = (
            ep.get("summary", "")
            or ep.get("summary_context", "")
            or ep.get("conversation_summary", "")
            or ep.get("sliced_context", "")
            or ""
        )

        summary = summary[-self.token_cap:]
        options = (ep.get("options", "") or "")[:self.options_cap_chars]
        question = ep.get("question", "") or ""

        prompt = (
            f"Conversation summary:\n{summary}\n\n"
            f"Question: {question}\n"
            f"Options:\n{options}\n\n"
            f"Output format must be EXACTLY one of: (a) (b) (c) (d). No other text."
        )

        raw_text = call_llm(prompt)
        pred = normalize_label(raw_text)

        stored_count = 1 if summary else 0
        superseded_count = 0

        return pred, raw_text, [], [], stored_count, superseded_count


class VanillaRAGMethod:
    """
    Vanilla RAG baseline wrapper that reuses the shared retrieval pipeline.
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


def build_output_path(method_name: str, out_dir: str, episodes_path: str) -> str:
    dataset_stem = os.path.splitext(os.path.basename(episodes_path))[0]
    return os.path.join(out_dir, f"{method_name}_{dataset_stem}.jsonl")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--method",
        choices=["no_memory", "summary_only", "vanilla_rag", "controlled"],
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
    parser.add_argument(
        "--offset",
        type=int,
        default=0,
        help="Start index in the episode file (0-based)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of episodes to run after offset",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Explicit output jsonl path. Overrides out_dir-based default.",
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    if args.method == "no_memory":
        method = NoMemoryMethod(k=args.k, token_cap=args.token_cap)
    elif args.method == "summary_only":
        method = SummaryOnlyMethod(k=args.k, token_cap=args.token_cap)
    elif args.method == "vanilla_rag":
        method = VanillaRAGMethod(k=args.k, token_cap=args.token_cap)
    else:
        from eval.methods.controlled import ControlledMethod

        method = ControlledMethod(
            run_root=os.path.join(args.out_dir, "controlled_runs"),
            top_k=args.k,
        )

    out_path = args.output or build_output_path(args.method, args.out_dir, args.episodes_path)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    correct = 0
    total = 0

    with open(out_path, "w", encoding="utf-8") as out:
        processed = 0

        for idx, ep in enumerate(iter_jsonl(args.episodes_path)):
            if idx < args.offset:
                continue

            if args.limit is not None and processed >= args.limit:
                break

            start = time.time()
            (
                pred,
                raw_text,
                retrieved_texts,
                retrieved_scores,
                stored_count,
                superseded_count,
            ) = method.answer(ep)
            elapsed = time.time() - start

            print(
                f"[eval] file_idx={idx} local_idx={processed} episode {ep.get('episode_id')} took {elapsed:.2f}s",
                flush=True,
            )

            gold = normalize_label(ep.get("answer", ""))
            correct_flag = is_correct(pred, gold)
            correct += int(correct_flag)
            total += 1
            processed += 1

            contradiction_count = 0

            if args.method == "vanilla_rag":
                stored_count = len(ep.get("turns", []))
            elif args.method == "summary_only":
                stored_count = 1 if (
                    ep.get("summary", "")
                    or ep.get("summary_context", "")
                    or ep.get("conversation_summary", "")
                    or ep.get("sliced_context", "")
                ) else 0
            elif args.method == "controlled" and getattr(method, "last_trace", None):
                stored_count = method.last_trace.get("stored_memories_count", stored_count)
                superseded_count = method.last_trace.get("superseded_count", superseded_count)
                contradiction_count = method.last_trace.get("contradiction_count", 0)

            record = {
                # Core eval info
                "episode_id": ep.get("episode_id"),
                "question_id": ep.get("question_id"),
                "shared_context_id": ep.get("shared_context_id"),
                "persona_id": ep.get("persona_id"),
                "method": args.method,
                "prediction_raw": raw_text,
                "prediction": pred,
                "gold_raw": ep.get("answer", ""),
                "gold": gold,
                "correct": correct_flag,

                # Slice bookkeeping
                "file_index": idx,
                "offset": args.offset,
                "limit": args.limit,

                # Retrieval / memory logs
                "retrieved": retrieved_texts,
                "retrieved_scores": retrieved_scores,
                "stored_count": stored_count,
                "retrieved_count": len(retrieved_texts),
                "superseded_count": superseded_count,
                "contradiction_count": contradiction_count,

                # Run settings
                "k": args.k,
                "token_cap": args.token_cap,

                # Scenario breakdown variables
                "question_type": ep.get("question_type"),
                "topic": ep.get("topic", ""),
                "context_length_in_tokens": ep.get("context_length_in_tokens"),
                "context_length_in_letters": ep.get("context_length_in_letters"),
                "distance_to_ref_in_blocks": ep.get("distance_to_ref_in_blocks"),
                "distance_to_ref_in_tokens": ep.get("distance_to_ref_in_tokens"),
                "num_irrelevant_tokens": ep.get("num_irrelevant_tokens"),
                "distance_to_ref_proportion_in_context": ep.get("distance_to_ref_proportion_in_context"),
                "end_index_in_shared_context": ep.get("end_index_in_shared_context"),
            }

            out.write(json.dumps(record) + "\n")

        print(
            f"Processed {total} episodes from offset={args.offset} limit={args.limit}",
            flush=True,
        )

    acc = correct / total if total else 0.0
    print(f"Accuracy: {correct}/{total} = {acc:.3f}")
    print(f"Wrote logs to: {out_path}")


if __name__ == "__main__":
    main()
