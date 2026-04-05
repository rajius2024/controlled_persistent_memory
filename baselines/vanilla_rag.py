"""
Vanilla RAG Baseline
====================
Lead: Aravindan Chidambaram

Imports shared utilities:
  - utils/llm.py      → call_llm(), normalize_label(), is_correct()
  - baselines/embeddings.py → embed(), cosine_similarity()

Log schema matches eval/run_eval.py (Harshini) exactly.

Usage:
    export GROQ_API_KEY="your_key_here"
    python baselines/vanilla_rag.py
"""

import json
import os
import time
from collections import defaultdict

import numpy as np

from baselines.embeddings import embed_one, cosine_similarity, top_k_indices, embed_with_cache
from utils.llm import call_llm, normalize_label, is_correct, sleep_between_calls

# ── Config ────────────────────────────────────────────────────────────────────
TOP_K          = 15
RECENCY_WEIGHT = 0.15
RESULTS_DIR    = "results"
METHOD_NAME    = "vanilla_rag"


# ── Retrieval ──────────────────────────────────────────────────────────────────

def retrieve_top_k(
    query: str,
    turns: list[dict],
    embeddings: np.ndarray,
    k: int = TOP_K,
    recency_weight: float = RECENCY_WEIGHT,
) -> list[dict]:
    """Retrieve top-k turns using cosine similarity + recency boost."""
    if embeddings is None or len(turns) == 0:
        return []
    # Safety: trim stale cache if lengths don't match
    if len(embeddings) != len(turns):
        embeddings = embeddings[:len(turns)]
    k          = min(k, len(turns))
    query_vec  = embed_one(query)
    sim_scores = cosine_similarity(query_vec, embeddings)

    indices      = np.array([t.get("turn_index", i) for i, t in enumerate(turns)], dtype=float)
    min_i, max_i = indices.min(), indices.max()
    recency      = (indices - min_i) / (max_i - min_i + 1e-10)
    combined     = (1 - recency_weight) * sim_scores + recency_weight * recency
    top_idx      = top_k_indices(combined, k=k)

    texts = [f"{t['role'].capitalize()}: {t['content']}" for t in turns]
    return [
        {
            "text":       texts[i],
            "sim_score":  float(sim_scores[i]),
            "combined":   float(combined[i]),
            "turn_index": int(indices[i]),
        }
        for i in top_idx
    ]


# ── Prompt Builder ─────────────────────────────────────────────────────────────

def build_prompt(episode: dict, retrieved: list[dict]) -> str:
    sorted_turns = sorted(retrieved, key=lambda x: x["turn_index"])
    context      = "\n".join(r["text"] for r in sorted_turns)
    return (
        f"Below is a relevant portion of a conversation with a user:\n\n"
        f"{context}\n\n"
        f"Based on this conversation, answer the following question about the user:\n"
        f"Question: {episode['question']}\n"
        f"Options:\n{episode['options']}\n\n"
        f"Output format must be EXACTLY one of: (a) (b) (c) (d). No other text."
    )


# ── Main Runner ────────────────────────────────────────────────────────────────

def run_vanilla_rag(episodes: list[dict], top_k: int = TOP_K) -> dict:
    """
    Run vanilla RAG on episodes.
    Saves to results/vanilla_rag_dev10.jsonl with Harshini's log schema.
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)
    results_path = os.path.join(RESULTS_DIR, "vanilla_rag_dev10.jsonl")

    correct     = 0
    all_results = []
    by_type     = defaultdict(list)

    for ep in episodes:
        turns = ep.get("turns", [])
        if not turns:
            print(f"[{ep['episode_id']}] No turns — skipping.")
            continue

        # 1. Embed all turns (with disk cache)
        texts      = [f"{t['role'].capitalize()}: {t['content']}" for t in turns]
        embeddings = embed_with_cache(texts, cache_key=f"ep_{ep['question_id']}")

        # 2. Retrieve top-k (question + options as query)
        enhanced_query = f"{ep['question']} {ep['options']}"
        retrieved      = retrieve_top_k(enhanced_query, turns, embeddings, k=top_k)

        # 3. Call LLM
        prompt   = build_prompt(ep, retrieved)
        raw_text = call_llm(prompt)
        pred     = normalize_label(raw_text)
        gold     = normalize_label(ep["answer"])
        correct_ = is_correct(pred, gold)
        correct  += int(correct_)

        q_type = ep.get("question_type", "unknown")
        by_type[q_type].append(correct_)

        # 4. Log record — matches Harshini's schema exactly
        record = {
            "episode_id":       ep["episode_id"],
            "question_id":      ep["question_id"],
            "method":           METHOD_NAME,
            "prediction_raw":   raw_text,
            "prediction":       pred,
            "gold_raw":         ep["answer"],
            "gold":             gold,
            "correct":          correct_,
            "retrieved":        [r["text"] for r in retrieved],
            "retrieved_scores": [r["sim_score"] for r in retrieved],
            "stored_count":     len(turns),
            "retrieved_count":  len(retrieved),
            "superseded_count": 0,
            "question_type":    q_type,
            "topic":            ep.get("topic", ""),
            "top_k":            top_k,
        }
        all_results.append(record)

        print(
            f"[{ep['episode_id']}] "
            f"{len(turns)} stored → {len(retrieved)} retrieved | "
            f"Pred: {raw_text!r} → {pred} | Gold: {gold} | "
            f"{'✅' if correct_ else '❌'}"
        )

        sleep_between_calls()

    # Save
    with open(results_path, "w") as f:
        for r in all_results:
            f.write(json.dumps(r) + "\n")
    print(f"\n[Saved] {results_path}")

    accuracy = correct / len(episodes) if episodes else 0.0
    print(f"\n{'='*52}")
    print(f"  VANILLA RAG  (top_k={top_k}, recency={RECENCY_WEIGHT})")
    print(f"{'='*52}")
    print(f"  Accuracy : {correct}/{len(episodes)} = {accuracy:.2%}")
    print(f"\n  By question type:")
    for qt, vals in sorted(by_type.items()):
        print(f"    {qt:<40} {sum(vals)}/{len(vals)} = {sum(vals)/len(vals):.2%}")
    print(f"{'='*52}\n")

    return {"method": METHOD_NAME, "top_k": top_k, "correct": correct,
            "total": len(episodes), "accuracy": accuracy,
            "by_type": {qt: sum(v)/len(v) for qt, v in by_type.items()}}


if __name__ == "__main__":
    episodes_path = os.path.join("data", "dev_latest.jsonl")
    print(f"Loading episodes from {episodes_path}...")
    episodes = []
    with open(episodes_path) as f:
        for line in f:
            episodes.append(json.loads(line))
    print(f"Loaded {len(episodes)} episodes.\n")
    run_vanilla_rag(episodes, top_k=TOP_K)
