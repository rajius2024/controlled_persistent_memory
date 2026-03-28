"""
Vanilla RAG Baseline
====================
Lead: Aravindan Chidambaram

What this does:
- Indexes ALL conversation turns from an episode into an in-memory vector store
- At query time, retrieves top-k most similar turns using cosine similarity
  with a recency boost (more recent turns ranked slightly higher)
- Passes retrieved turns + question to the LLM (no versioning, no filtering)
- This is the "store-all" baseline our CPM system is compared against

Key difference from CPM:
- No write filtering  : stores everything, not just preference facts
- No versioning       : outdated preferences are NOT marked superseded
- No intent awareness : retrieval is purely similarity-based

Usage:
    export GROQ_API_KEY="your_key_here"
    python baselines/vanilla_rag.py
"""

import json
import os
import re
import time
from collections import defaultdict

import numpy as np
from groq import Groq
from sentence_transformers import SentenceTransformer

# ── Config ────────────────────────────────────────────────────────────────────
EMBEDDING_MODEL  = "all-MiniLM-L6-v2"
TOP_K            = 15     # retrieve more turns for better coverage
RECENCY_WEIGHT   = 0.15   # small boost for more recent turns (0 = pure similarity)
MAX_RETRIES      = 3
RATE_LIMIT_SLEEP = 2.0    # seconds between requests
RESULTS_DIR      = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")

print("Loading embedding model...")
encoder = SentenceTransformer(EMBEDDING_MODEL)
client  = Groq(api_key=os.environ.get("GROQ_API_KEY"))


# ── LLM Call ──────────────────────────────────────────────────────────────────

def call_llm(prompt: str) -> str:
    """Call Llama 3.3-70b via Groq with retry on rate limit errors."""
    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a helpful assistant answering multiple-choice questions "
                            "about a user based on their conversation history. "
                            "Reply with ONLY the single letter of the correct option (a, b, c, or d). "
                            "No punctuation, no explanation."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                max_tokens=5,
                temperature=0.0,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            err = str(e)
            if "413" in err or "rate_limit" in err.lower() or "429" in err:
                wait = 15 * (attempt + 1)
                print(f"  [RATE LIMIT] Waiting {wait}s... (attempt {attempt+1}/{MAX_RETRIES})")
                time.sleep(wait)
            elif "401" in err:
                print("  [AUTH ERROR] Invalid GROQ_API_KEY. Set: export GROQ_API_KEY=...")
                return ""
            else:
                print(f"  [LLM ERROR] {e}")
                return ""
    print(f"  [FAILED] All {MAX_RETRIES} retries exhausted.")
    return ""


# ── Answer Parsing ─────────────────────────────────────────────────────────────

def parse_answer(text: str) -> str:
    """Strip everything except a/b/c/d from an answer string."""
    return re.sub(r'[^a-d]', '', text.strip().lower())[:1]


# ── In-Memory Vector Store ─────────────────────────────────────────────────────

class InMemoryVectorStore:
    """
    Lightweight in-memory vector store using numpy cosine similarity.
    Supports recency-weighted re-ranking so recent turns get a small boost.
    """

    def __init__(self):
        self.embeddings:   np.ndarray = None
        self.texts:        list       = []
        self.turn_indices: list       = []

    def index(self, turns: list):
        """Embed all turns. Each turn dict must have: role, content, turn_index."""
        self.texts = [
            f"{t['role'].capitalize()}: {t['content']}"
            for t in turns
        ]
        self.turn_indices = [t.get("turn_index", i) for i, t in enumerate(turns)]
        self.embeddings   = encoder.encode(self.texts, show_progress_bar=False)

    def retrieve(self, query: str, k: int = TOP_K, recency_weight: float = RECENCY_WEIGHT) -> list:
        """
        Retrieve top-k turns using cosine similarity + recency boost.

        Recency score = normalised turn_index (later turn → higher score).
        Final score   = (1 - recency_weight) * similarity + recency_weight * recency
        """
        if self.embeddings is None or not self.texts:
            return []

        k = min(k, len(self.texts))

        # Cosine similarity
        q_vec  = encoder.encode([query])[0]
        q_norm = q_vec / (np.linalg.norm(q_vec) + 1e-10)
        m_norm = self.embeddings / (
            np.linalg.norm(self.embeddings, axis=1, keepdims=True) + 1e-10
        )
        sim_scores = m_norm @ q_norm

        # Recency scores — normalise turn indices to [0, 1]
        indices      = np.array(self.turn_indices, dtype=float)
        min_i, max_i = indices.min(), indices.max()
        recency_scores = (
            (indices - min_i) / (max_i - min_i) if max_i > min_i
            else np.ones_like(indices)
        )

        combined    = (1 - recency_weight) * sim_scores + recency_weight * recency_scores
        top_indices = np.argsort(combined)[::-1][:k]

        return [
            {
                "text":       self.texts[i],
                "sim_score":  float(sim_scores[i]),
                "combined":   float(combined[i]),
                "turn_index": self.turn_indices[i],
            }
            for i in top_indices
        ]


# ── Prompt Builder ─────────────────────────────────────────────────────────────

def build_prompt(episode: dict, retrieved: list) -> str:
    """
    Build the LLM prompt.
    Turns are sorted chronologically so the model reads them in order.
    """
    sorted_turns = sorted(retrieved, key=lambda x: x["turn_index"])
    context      = "\n".join(r["text"] for r in sorted_turns)

    return (
        f"Below is a relevant portion of a conversation with a user:\n\n"
        f"{context}\n\n"
        f"Based on this conversation, answer the following question about the user:\n"
        f"Question: {episode['question']}\n"
        f"Options:\n{episode['options']}\n\n"
        f"Reply with only the letter (a, b, c, or d)."
    )


# ── Main Runner ────────────────────────────────────────────────────────────────

def run_vanilla_rag(episodes: list, top_k: int = TOP_K) -> dict:
    """
    Run vanilla RAG on a list of episodes.
    Saves per-episode results to results/vanilla_rag_results.jsonl.
    Returns aggregated metrics dict.
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)
    results_path = os.path.join(RESULTS_DIR, "vanilla_rag_results.jsonl")

    correct     = 0
    all_results = []
    store       = InMemoryVectorStore()
    by_type     = defaultdict(list)

    for ep in episodes:
        turns = ep.get("turns", [])
        if not turns:
            print(f"[{ep['episode_id']}] ⚠️  No turns — skipping.")
            continue

        # 1. Enhanced query: question + options → better retrieval signal
        enhanced_query = f"{ep['question']} {ep['options']}"

        # 2. Index ALL turns (no filter — this is the vanilla store-all approach)
        store.index(turns)

        # 3. Retrieve top-k with recency boost
        retrieved      = store.retrieve(enhanced_query, k=top_k)
        token_estimate = sum(len(r["text"]) for r in retrieved) // 4

        # 4. Build prompt and call LLM
        prompt   = build_prompt(ep, retrieved)
        response = call_llm(prompt)

        # 5. Parse and compare
        predicted      = parse_answer(response)
        correct_letter = parse_answer(ep["answer"])
        is_correct     = predicted == correct_letter
        correct        += int(is_correct)

        q_type = ep.get("question_type", "unknown")
        by_type[q_type].append(is_correct)

        # 6. Per-episode result record
        result = {
            "episode_id":      ep["episode_id"],
            "question_id":     ep["question_id"],
            "question_type":   q_type,
            "predicted":       predicted,
            "correct":         correct_letter,
            "is_correct":      is_correct,
            "turns_indexed":   len(turns),
            "turns_retrieved": len(retrieved),
            "token_estimate":  token_estimate,
            "top_sim_score":   retrieved[0]["sim_score"] if retrieved else 0.0,
        }
        all_results.append(result)

        print(
            f"[{ep['episode_id']}] "
            f"{len(turns)} turns → top {len(retrieved)} (~{token_estimate} tok) | "
            f"Pred: {response.strip()!r} | Ans: {ep['answer']} | "
            f"{'✅' if is_correct else '❌'}"
        )

        time.sleep(RATE_LIMIT_SLEEP)

    # 7. Save results
    with open(results_path, "w") as f:
        for r in all_results:
            f.write(json.dumps(r) + "\n")
    print(f"\n[Saved] Results → {results_path}")

    # 8. Print summary
    accuracy = correct / len(episodes) if episodes else 0.0

    print(f"\n{'='*52}")
    print(f"  VANILLA RAG RESULTS  (top_k={top_k}, recency={RECENCY_WEIGHT})")
    print(f"{'='*52}")
    print(f"  Overall Accuracy : {correct}/{len(episodes)} = {accuracy:.2%}")
    print(f"\n  Breakdown by question type:")
    for qt, vals in sorted(by_type.items()):
        acc = sum(vals) / len(vals)
        print(f"    {qt:<38} {sum(vals)}/{len(vals)} = {acc:.2%}")
    print(f"{'='*52}\n")

    return {
        "method":   "vanilla_rag",
        "top_k":    top_k,
        "correct":  correct,
        "total":    len(episodes),
        "accuracy": accuracy,
        "by_type":  {qt: sum(v)/len(v) for qt, v in by_type.items()},
    }


# ── Entry Point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    episodes_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "data", "dev_10.jsonl"
    )
    print(f"Loading episodes from {episodes_path}...")
    episodes = []
    with open(episodes_path) as f:
        for line in f:
            episodes.append(json.loads(line))
    print(f"Loaded {len(episodes)} episodes.\n")

    run_vanilla_rag(episodes, top_k=TOP_K)
