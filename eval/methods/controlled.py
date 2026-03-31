from __future__ import annotations

import shutil
from hashlib import sha1
from pathlib import Path
from typing import Any

from baselines.embeddings import (
    cosine_similarity,
    embed_one,
    embed_with_cache,
    top_k_indices,
)
from memory.chroma_store import ChromaMemoryStore
from memory.controlled_store_builder import build_controlled_store_from_episode
from memory.logging_utils import log_episode_summary, log_retrieval_event
from memory.schema import MemoryRecord, MemoryStatus
from utils.llm import call_llm, normalize_label


class ControlledMethod:
    """
    Controlled-memory method with persistent storage and active-only retrieval.
    """

    def __init__(
        self,
        run_root: str,
        top_k: int = 15,
        memory_type: str | None = None,
        reset_episode_run_dir: bool = True,
    ) -> None:
        self.run_root = Path(run_root)
        self.run_root.mkdir(parents=True, exist_ok=True)

        self.top_k = top_k
        self.memory_type = memory_type
        self.reset_episode_run_dir = reset_episode_run_dir
        self.last_trace: dict[str, Any] | None = None

    def answer(self, ep: dict):
        """
        Return the runner-compatible tuple:
        prediction, raw model output, retrieved texts, retrieved scores
        """
        trace = self.answer_with_trace(ep)
        self.last_trace = trace

        pred = normalize_label(trace["prediction_raw"])
        raw_text = trace["prediction_raw"]
        retrieved_texts = trace["retrieved_memories"]
        retrieved_scores = trace["retrieved_scores"]

        return pred, raw_text, retrieved_texts, retrieved_scores

    def answer_with_trace(self, episode: dict[str, Any]) -> dict[str, Any]:
        episode_id = episode["episode_id"]
        persona_id = str(episode["persona_id"])

        episode_run_dir = self.run_root / f"episode_{episode_id}"

        # Rebuild per-episode state to avoid stale persisted records across reruns.
        if self.reset_episode_run_dir and episode_run_dir.exists():
            shutil.rmtree(episode_run_dir)

        store = ChromaMemoryStore(
            run_dir=str(episode_run_dir),
            collection_name=f"episode_{episode_id}_memory",
        )

        build_stats = build_controlled_store_from_episode(
            episode=episode,
            store=store,
            run_dir=str(episode_run_dir),
        )

        active_memories = store.list_memories(
            persona_id=persona_id,
            status=MemoryStatus.ACTIVE.value,
            memory_type=self.memory_type,
        )

        retrieved_records, retrieved_scores = self._retrieve_top_k(
            episode=episode,
            memories=active_memories,
            k=self.top_k,
        )

        retrieved_texts = [m.normalized_mem_text for m in retrieved_records]

        prompt = self._build_prompt(episode, retrieved_texts)
        raw_text = call_llm(prompt)

        trace = {
            "episode_id": episode_id,
            "question_id": episode.get("question_id"),
            "persona_id": persona_id,
            "prediction_raw": raw_text,
            "prediction": normalize_label(raw_text),
            "gold": episode.get("answer"),
            "stored_memories_count": build_stats["stored_count"],
            "active_memories_count": len(active_memories),
            "active_memories": [m.normalized_mem_text for m in active_memories],
            "retrieved_memories": retrieved_texts,
            "retrieved_scores": retrieved_scores,
            "retrieved_count": len(retrieved_texts),
            "superseded_count": build_stats["superseded_count"],
            "superseded_ids": build_stats["superseded_ids"],
            "top_k": self.top_k,
            "question_type": episode.get("question_type"),
            "topic": episode.get("topic", ""),
        }

        log_retrieval_event(
            str(episode_run_dir),
            {
                "episode_id": episode_id,
                "question_id": episode.get("question_id"),
                "persona_id": persona_id,
                "query_text": episode["question"],
                "top_k": self.top_k,
                "active_memory_count": len(active_memories),
                "active_memory_ids": [m.memory_id for m in active_memories],
                "retrieved_memory_ids": [m.memory_id for m in retrieved_records],
                "retrieved_texts": retrieved_texts,
                "retrieved_scores": retrieved_scores,
                "retrieved_count": len(retrieved_texts),
            },
        )

        log_episode_summary(str(episode_run_dir), trace)
        self.last_trace = trace
        return trace

    def _retrieve_top_k(
        self,
        episode: dict[str, Any],
        memories: list[MemoryRecord],
        k: int,
    ) -> tuple[list[MemoryRecord], list[float]]:
        if not memories:
            return [], []

        texts = [m.normalized_mem_text for m in memories]

        # Include a content fingerprint so embedding cache entries track the
        # active memory set instead of only the episode identifier.
        fingerprint = sha1("||".join(texts).encode("utf-8")).hexdigest()[:16]
        cache_key = f"controlled_ep_{episode.get('question_id', episode.get('episode_id'))}_{fingerprint}"

        matrix = embed_with_cache(texts, cache_key=cache_key)

        query = f"{episode.get('question', '')} {episode.get('options', '')}"
        query_vec = embed_one(query)

        scores = cosine_similarity(query_vec, matrix)
        top_idx = top_k_indices(scores, k=min(k, len(memories)))

        retrieved_records = [memories[i] for i in top_idx]
        retrieved_scores = [float(scores[i]) for i in top_idx]

        return retrieved_records, retrieved_scores

    def _build_prompt(self, episode: dict[str, Any], retrieved_texts: list[str]) -> str:
        memories_block = "\n".join(f"- {text}" for text in retrieved_texts) if retrieved_texts else "None"

        question = episode.get("question", "") or ""
        options = episode.get("options", "") or ""

        return (
            f"Below are the user's currently active memories:\n\n"
            f"{memories_block}\n\n"
            f"Based on these active memories, answer the following question about the user:\n"
            f"Question: {question}\n"
            f"Options:\n{options}\n\n"
            f"Output format must be EXACTLY one of: (a) (b) (c) (d). No other text."
        )
