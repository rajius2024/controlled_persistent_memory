from __future__ import annotations

import re
import math
import shutil
from collections import Counter
from hashlib import sha1
from pathlib import Path
from typing import Any

from baselines.embeddings import (
    cosine_similarity,
    embed_one,
    embed_with_cache,
)
from memory.chroma_store import ChromaMemoryStore
from memory.controlled_store_builder import build_controlled_store_from_episode
from memory.logging_utils import log_episode_summary, log_retrieval_event
from memory.schema import MemoryRecord, MemoryStatus
from utils.llm import call_llm, normalize_label


GENERIC_MEMORY_PATTERNS = {
    "likes music",
    "interested in music",
    "appreciates literature",
    "interested in literature",
    "likes books",
    "likes reading",
}

GENERIC_TOKENS = {
    "music",
    "books",
    "book",
    "reading",
    "literature",
    "art",
    "stories",
    "storytelling",
}

EVOLUTION_QUESTION_TYPES = {
    "track_full_preference_evolution",
    "recalling_the_reasons_behind_previous_updates",
}


class ControlledMethod:
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
        trace = self.answer_with_trace(ep)
        self.last_trace = trace

        pred = normalize_label(trace["prediction_raw"])
        raw_text = trace["prediction_raw"]
        retrieved_texts = trace["retrieved_memories"]
        retrieved_scores = trace["retrieved_scores"]

        stored_count = trace.get("stored_memories_count", 0)
        superseded_count = trace.get("superseded_count", 0)

        return pred, raw_text, retrieved_texts, retrieved_scores, stored_count, superseded_count

    def answer_with_trace(self, episode: dict[str, Any]) -> dict[str, Any]:
        episode_id = episode["episode_id"]
        persona_id = str(episode["persona_id"])

        episode_run_dir = self.run_root / f"episode_{episode_id}"

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

        question_type = episode.get("question_type", "")
        if question_type in EVOLUTION_QUESTION_TYPES:
            superseded_memories = store.list_memories(
                persona_id=persona_id,
                status=MemoryStatus.SUPERSEDED.value,
                memory_type=self.memory_type,
            )
        else:
            superseded_memories = []

        retrieved_records, retrieved_scores, retrieval_debug = self._retrieve_top_k(
            episode=episode,
            active_memories=active_memories,
            superseded_memories=superseded_memories,
            k=self.top_k,
        )

        retrieved_texts = [m.normalized_mem_text for m in retrieved_records]

        prompt = self._build_prompt(episode, retrieved_records)
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
            "contradiction_count": build_stats["contradiction_count"],
            "top_k": self.top_k,
            "question_type": question_type,
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
                "retrieval_debug": retrieval_debug,
                "retrieved_count": len(retrieved_texts),
            },
        )

        log_episode_summary(str(episode_run_dir), trace)
        self.last_trace = trace
        return trace

    def _infer_query_slot(self, question: str, options: str) -> str | None:
        combined = f"{question} {options}".lower()

        if any(token in combined for token in [
            "music", "song", "playlist", "genre", "beats", "jazz", "rock", "pop",
            "album", "artist", "artists", "documentary", "podcast", "instrument",
            "instruments", "rhythm", "melody"
        ]):
            return "music"

        if any(token in combined for token in [
            "book", "books", "reading", "read", "reader", "novel", "novels",
            "literature", "literary", "author", "authors", "bookstore",
            "bookstores", "library", "libraries", "blog", "blogging",
            "fiction", "nonfiction", "poetry"
        ]):
            return "books"

        if any(token in combined for token in ["coffee", "tea", "juice", "drink", "soda"]):
            return "drink"

        if any(token in combined for token in ["pizza", "pasta", "spicy", "food", "breakfast", "dinner", "cuisine"]):
            return "food"

        if any(token in combined for token in ["text", "phone", "call", "email"]):
            return "communication"

        if any(token in combined for token in ["morning", "night", "weekend", "weekday"]):
            return "schedule"

        return None

    def _memory_tokens(self, text: str) -> set[str]:
        return set(re.findall(r"[a-z0-9]+", text.lower()))

    def _parse_options(self, options: str) -> list[str]:
        matches = re.findall(
            r"\([a-d]\)\s*(.*?)(?=\s*\([a-d]\)|$)",
            options,
            flags=re.IGNORECASE | re.DOTALL,
        )
        return [m.strip() for m in matches if m.strip()]


    def _bm25_retrieve(
        self,
        memories: list[MemoryRecord],
        query: str,
        k: int = 20,
    ) -> list[MemoryRecord]:
        if not memories:
            return []

        query_terms = list(self._memory_tokens(query))
        if not query_terms:
            return []

        docs: list[list[str]] = []
        doc_freq: Counter[str] = Counter()

        for memory in memories:
            tokens = list(self._memory_tokens(memory.normalized_mem_text))
            docs.append(tokens)
            for term in set(tokens):
                doc_freq[term] += 1

        N = len(docs)
        avgdl = sum(len(doc) for doc in docs) / max(N, 1)

        k1 = 1.5
        b = 0.75

        scored: list[tuple[float, MemoryRecord]] = []

        for memory, doc in zip(memories, docs):
            doc_len = len(doc)
            tf = Counter(doc)
            score = 0.0

            for term in query_terms:
                if term not in tf:
                    continue

                df = doc_freq.get(term, 0)
                idf = math.log(1 + (N - df + 0.5) / (df + 0.5))
                freq = tf[term]

                denom = freq + k1 * (1 - b + b * doc_len / max(avgdl, 1e-9))
                score += idf * (freq * (k1 + 1)) / max(denom, 1e-9)

            if score > 0.0:
                scored.append((score, memory))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [memory for _, memory in scored[:k]]


    def _option_conditioned_candidates(
        self,
        memories: list[MemoryRecord],
        question: str,
        options: str,
        per_option_k: int = 3,
    ) -> list[MemoryRecord]:
        option_list = self._parse_options(options)
        if not option_list or not memories:
            return []

        selected: list[MemoryRecord] = []
        seen_ids: set[str] = set()

        for option in option_list:
            option_query = f"{question} {option}"
            rows = self._scored_candidates(
                memories,
                query=option_query,
                cache_key_prefix=f"controlled_option_{sha1(option_query.encode('utf-8')).hexdigest()[:8]}",
            )
            rows = sorted(rows, key=lambda r: r["score"], reverse=True)[:per_option_k]

            for row in rows:
                memory = row["memory"]
                if memory.memory_id in seen_ids:
                    continue
                selected.append(memory)
                seen_ids.add(memory.memory_id)

        return selected


    def _specificity_bonus(self, memory: MemoryRecord) -> float:
        text = memory.normalized_mem_text.lower()
        tokens = self._memory_tokens(text)

        bonus = 0.0
        bonus += min(len(tokens), 8) * 0.015

        if any(word in text for word in ["producing", "creates", "started", "hosts", "joined", "volunteers", "visits", "uses"]):
            bonus += 0.08

        target_kind = memory.metadata.get("target_kind")
        if target_kind == "routine":
            bonus += 0.05
        if target_kind == "profile":
            bonus += 0.04

        if any(word in text for word in ["electronic", "pacific", "classic", "indie", "original", "instruments"]):
            bonus += 0.05

        return bonus

    def _generic_penalty(self, memory: MemoryRecord) -> float:
        text = memory.normalized_mem_text.lower()
        tokens = self._memory_tokens(text)

        penalty = 0.0

        if text in GENERIC_MEMORY_PATTERNS:
            penalty += 0.18

        generic_count = sum(1 for tok in tokens if tok in GENERIC_TOKENS)
        if generic_count >= 1 and len(tokens) <= 3:
            penalty += 0.10

        if text.startswith(("likes ", "interested in ", "appreciates ")) and len(tokens) <= 4:
            penalty += 0.06

        return penalty

    def _rerank_score(self, semantic_score: float, memory: MemoryRecord) -> float:
        specificity = float(memory.metadata.get("specificity_score", 0.0))
        return semantic_score + self._specificity_bonus(memory) - self._generic_penalty(memory) + 0.15 * specificity

    def _scored_candidates(
        self,
        memories: list[MemoryRecord],
        query: str,
        cache_key_prefix: str,
    ) -> list[dict[str, Any]]:
        if not memories:
            return []

        texts = [m.normalized_mem_text for m in memories]
        fingerprint = sha1("||".join(texts).encode("utf-8")).hexdigest()[:16]
        cache_key = f"{cache_key_prefix}_{fingerprint}"

        matrix = embed_with_cache(texts, cache_key=cache_key)
        query_vec = embed_one(query)
        semantic_scores = cosine_similarity(query_vec, matrix)

        rows: list[dict[str, Any]] = []
        for i, memory in enumerate(memories):
            semantic = float(semantic_scores[i])
            final = self._rerank_score(semantic, memory)
            rows.append(
                {
                    "memory": memory,
                    "score": final,
                    "semantic_score": semantic,
                    "status": str(memory.status),
                    "polarity": memory.metadata.get("polarity"),
                    "target_kind": memory.metadata.get("target_kind"),
                }
            )
        return rows

    def _retrieve_top_k(
        self,
        episode: dict[str, Any],
        active_memories: list[MemoryRecord],
        superseded_memories: list[MemoryRecord],
        k: int,
    ) -> tuple[list[MemoryRecord], list[float], list[dict[str, Any]]]:
        if not active_memories and not superseded_memories:
            return [], [], []

        query_slot = self._infer_query_slot(
            question=episode.get("question", ""),
            options=episode.get("options", ""),
        )

        filtered_active = active_memories
        filtered_superseded = superseded_memories

        if query_slot is not None:
            slot_active = [m for m in active_memories if m.metadata.get("slot") == query_slot]
            if len(slot_active) >= 3:
                filtered_active = slot_active
            else:
                filtered_active = active_memories

            slot_superseded = [m for m in superseded_memories if m.metadata.get("slot") == query_slot]
            if len(slot_superseded) >= 2:
                filtered_superseded = slot_superseded
            else:
                filtered_superseded = superseded_memories

        query = f"{episode.get('question', '')} {episode.get('options', '')}"
        question_type = episode.get("question_type", "")
        effective_k = self.top_k
        if question_type in EVOLUTION_QUESTION_TYPES:
            effective_k = max(self.top_k, 7)

        dense_pool = filtered_active

        bm25_records = self._bm25_retrieve(
            dense_pool,
            query=query,
            k=20,
        )

        option_records = self._option_conditioned_candidates(
            dense_pool,
            question=episode.get("question", ""),
            options=episode.get("options", ""),
            per_option_k=3,
        )

        union_memories: list[MemoryRecord] = []
        seen_ids: set[str] = set()

        for memory in list(dense_pool) + bm25_records + option_records:
            if memory.memory_id in seen_ids:
                continue
            union_memories.append(memory)
            seen_ids.add(memory.memory_id)

        active_rows = self._scored_candidates(
            union_memories,
            query=query,
            cache_key_prefix=f"controlled_active_union_{episode.get('question_id', episode.get('episode_id'))}",
        )

        superseded_rows = self._scored_candidates(
            filtered_superseded,
            query=query,
            cache_key_prefix=f"controlled_superseded_{episode.get('question_id', episode.get('episode_id'))}",
        )

        if question_type in EVOLUTION_QUESTION_TYPES:
            pos_active = [r for r in active_rows if r["polarity"] == "positive"]
            neg_active = [r for r in active_rows if r["polarity"] == "negative"]
            neutral_active = [r for r in active_rows if r["polarity"] not in {"positive", "negative"}]

            pos_active.sort(key=lambda r: r["score"], reverse=True)
            neg_active.sort(key=lambda r: r["score"], reverse=True)
            neutral_active.sort(key=lambda r: r["score"], reverse=True)
            superseded_rows.sort(key=lambda r: r["score"], reverse=True)

            chosen_rows: list[dict[str, Any]] = []

            # Force some trajectory shape.
            if pos_active:
                chosen_rows.append(pos_active[0])
            if neg_active:
                chosen_rows.append(neg_active[0])
            if superseded_rows:
                chosen_rows.append(superseded_rows[0])

            seen_ids = {r["memory"].memory_id for r in chosen_rows}

            remaining = sorted(
                active_rows + superseded_rows,
                key=lambda r: r["score"],
                reverse=True,
            )
            for row in remaining:
                if row["memory"].memory_id in seen_ids:
                    continue
                chosen_rows.append(row)
                seen_ids.add(row["memory"].memory_id)
                if len(chosen_rows) >= min(effective_k, len(active_rows) + len(superseded_rows)):
                    break
        else:
            all_rows = sorted(active_rows, key=lambda r: r["score"], reverse=True)
            chosen_rows = all_rows[: min(effective_k, len(all_rows))]

        if len(chosen_rows) < 5:
            fallback_rows = self._scored_candidates(
                active_memories,
                query=query,
                cache_key_prefix=f"controlled_fallback_{episode.get('question_id', episode.get('episode_id'))}",
            )
            fallback_rows = sorted(fallback_rows, key=lambda r: r["score"], reverse=True)

            seen_ids = {row["memory"].memory_id for row in chosen_rows}
            merged_rows = list(chosen_rows)

            for row in fallback_rows:
                mem_id = row["memory"].memory_id
                if mem_id in seen_ids:
                    continue
                merged_rows.append(row)
                seen_ids.add(mem_id)
                if len(merged_rows) >= effective_k:
                    break

            chosen_rows = merged_rows[: min(effective_k, len(merged_rows))]


        retrieved_records = [row["memory"] for row in chosen_rows]
        retrieved_scores = [float(row["score"]) for row in chosen_rows]

        retrieval_debug = []
        all_debug_rows = active_rows + superseded_rows
        all_debug_rows.sort(key=lambda r: r["score"], reverse=True)

        for row in all_debug_rows:
            memory = row["memory"]
            retrieval_debug.append(
                {
                    "memory_id": memory.memory_id,
                    "text": memory.normalized_mem_text,
                    "score": float(row["score"]),
                    "semantic_score": float(row["semantic_score"]),
                    "slot": memory.metadata.get("slot"),
                    "polarity": memory.metadata.get("polarity"),
                    "target_kind": memory.metadata.get("target_kind"),
                    "status": str(memory.status),
                    "write_score": memory.metadata.get("write_score"),
                }
            )

        return retrieved_records, retrieved_scores, retrieval_debug

    def _build_prompt(self, episode: dict[str, Any], retrieved_records: list[MemoryRecord]) -> str:
        if retrieved_records:
            memory_lines = []
            for memory in retrieved_records:
                canonical = memory.normalized_mem_text
                provenance = str(memory.metadata.get("provenance_text", "")).strip()
                if provenance:
                    memory_lines.append(
                        f"- Memory: {canonical}\n"
                        f"  Evidence: {provenance}"
                    )
                else:
                    memory_lines.append(f"- Memory: {canonical}")
            memories_block = "\n".join(memory_lines)
        else:
            memories_block = "None"

        question = episode.get("question", "") or ""
        options = episode.get("options", "") or ""

        return (
            f"Below are the user's relevant memories and supporting evidence:\n\n"
            f"{memories_block}\n\n"
            f"Use both the memory statements and their evidence to answer the question.\n"
            f"Question: {question}\n"
            f"Options:\n{options}\n\n"
            f"Output format must be EXACTLY one of: (a) (b) (c) (d). No other text."
        )


