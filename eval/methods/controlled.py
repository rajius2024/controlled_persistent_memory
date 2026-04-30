from __future__ import annotations

import math
import os
import re
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

        self.ablation = os.environ.get("CONTROLLED_ABLATION", "full").strip().lower()
        valid_ablations = {"full", "no_bm25", "no_options", "no_superseded"}
        if self.ablation not in valid_ablations:
            raise ValueError(
                f"Unsupported CONTROLLED_ABLATION={self.ablation!r}. "
                f"Expected one of {sorted(valid_ablations)}."
            )
        print(f"[controlled] ablation={self.ablation}", flush=True)

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
        if (
            question_type in EVOLUTION_QUESTION_TYPES
            and self.ablation != "no_superseded"
        ):
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

        retrieved_texts = [self._retrieval_text(m) for m in retrieved_records]

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
            "ablation": self.ablation,
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

    def _retrieval_text(self, memory: MemoryRecord) -> str:
        canonical = memory.normalized_mem_text
        provenance = str(memory.metadata.get("provenance_text", "")).strip()
        status = str(memory.status)
        if provenance:
            return f"{canonical}. Status: {status}. Evidence: {provenance}"
        return f"{canonical}. Status: {status}"

    def _query_focus_mode(self, episode: dict[str, Any]) -> str:
        qtype = episode.get("question_type", "") or ""
        if qtype == "recall_user_shared_facts":
            return "raw_recall"
        if qtype in EVOLUTION_QUESTION_TYPES:
            return "timeline"
        if qtype in {"bookRecommendation", "musicRecommendation"}:
            return "slot_pref"
        return "default"

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

    def _evidence_bonus(self, memory: MemoryRecord, query: str) -> float:
        provenance = str(memory.metadata.get("provenance_text", "")).lower()
        if not provenance:
            return 0.0

        query_tokens = self._memory_tokens(query)
        prov_tokens = self._memory_tokens(provenance)
        if not query_tokens or not prov_tokens:
            return 0.0

        overlap = len(query_tokens & prov_tokens) / max(len(query_tokens), 1)
        return 0.12 * overlap

    def _temporal_score(self, memory: MemoryRecord, episode: dict[str, Any]) -> float:
        qtype = episode.get("question_type", "") or ""
        text = memory.normalized_mem_text.lower()
        status = str(memory.status)

        score = 0.0

        if qtype in EVOLUTION_QUESTION_TYPES:
            if "superseded" in status.lower():
                score += 0.08
            if memory.metadata.get("update_hint", False):
                score += 0.08
        else:
            if "active" in status.lower():
                score += 0.05
            if "superseded" in status.lower():
                score -= 0.03

        if any(phrase in text for phrase in ["used to", "no longer", "stopped", "not anymore"]):
            if qtype in EVOLUTION_QUESTION_TYPES:
                score += 0.08
            else:
                score -= 0.04

        return score

    def _rerank_score(self, semantic_score: float, memory: MemoryRecord, query: str, episode: dict[str, Any]) -> float:
        specificity = float(memory.metadata.get("specificity_score", 0.0))
        return (
            semantic_score
            + self._specificity_bonus(memory)
            - self._generic_penalty(memory)
            + 0.15 * specificity
            + self._evidence_bonus(memory, query)
            + self._temporal_score(memory, episode)
        )

    def _bm25_retrieve(
        self,
        memories: list[MemoryRecord],
        query: str,
        k: int = 20,
    ) -> list[MemoryRecord]:
        if not memories:
            return []

        docs: list[list[str]] = []
        doc_freq: Counter[str] = Counter()
        query_terms = list(self._memory_tokens(query))
        if not query_terms:
            return []

        for memory in memories:
            tokens = list(self._memory_tokens(self._retrieval_text(memory)))
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

        scored.sort(key=lambda x: (x[0], x[1].memory_id), reverse=True)
        return [memory for _, memory in scored[:k]]

    def _scored_candidates(
        self,
        memories: list[MemoryRecord],
        query: str,
        cache_key_prefix: str,
        episode: dict[str, Any],
    ) -> list[dict[str, Any]]:
        if not memories:
            return []

        texts = [self._retrieval_text(m) for m in memories]
        fingerprint = sha1("||".join(texts).encode("utf-8")).hexdigest()[:16]
        cache_key = f"{cache_key_prefix}_{fingerprint}"

        matrix = embed_with_cache(texts, cache_key=cache_key)
        query_vec = embed_one(query)
        semantic_scores = cosine_similarity(query_vec, matrix)

        rows: list[dict[str, Any]] = []
        for i, memory in enumerate(memories):
            semantic = float(semantic_scores[i])
            final = self._rerank_score(semantic, memory, query, episode)
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

    def _option_conditioned_rows(
        self,
        memories: list[MemoryRecord],
        question: str,
        options: str,
        episode: dict[str, Any],
        per_option_k: int = 4,
    ) -> list[dict[str, Any]]:
        option_list = self._parse_options(options)
        if not option_list or not memories:
            return []

        out: list[dict[str, Any]] = []
        seen: set[str] = set()

        for option in option_list:
            option_query = f"{question} {option}"
            rows = self._scored_candidates(
                memories,
                query=option_query,
                cache_key_prefix=f"controlled_option_{sha1(option_query.encode('utf-8')).hexdigest()[:8]}",
                episode=episode,
            )
            rows = sorted(rows, key=lambda r: (r["score"], r["memory"].memory_id), reverse=True)[:per_option_k]

            for row in rows:
                mem_id = row["memory"].memory_id
                if mem_id in seen:
                    continue
                out.append(row)
                seen.add(mem_id)

        return out

    def _rrf_fuse(
        self,
        ranked_lists: list[list[dict[str, Any]]],
        k: int = 60,
    ) -> list[dict[str, Any]]:
        fused: dict[str, dict[str, Any]] = {}

        for ranked in ranked_lists:
            for rank, row in enumerate(ranked, start=1):
                mem = row["memory"]
                mem_id = mem.memory_id
                score_add = 1.0 / (k + rank)

                if mem_id not in fused:
                    fused[mem_id] = {
                        "memory": mem,
                        "rrf_score": 0.0,
                        "best_semantic_score": float(row.get("semantic_score", 0.0)),
                        "best_score": float(row.get("score", 0.0)),
                        "status": row.get("status"),
                        "polarity": row.get("polarity"),
                        "target_kind": row.get("target_kind"),
                    }

                fused[mem_id]["rrf_score"] += score_add
                fused[mem_id]["best_semantic_score"] = max(
                    fused[mem_id]["best_semantic_score"],
                    float(row.get("semantic_score", 0.0)),
                )
                fused[mem_id]["best_score"] = max(
                    fused[mem_id]["best_score"],
                    float(row.get("score", 0.0)),
                )

        out = []
        for _, row in fused.items():
            out.append(
                {
                    "memory": row["memory"],
                    "score": float(row["rrf_score"]) + 0.15 * float(row["best_score"]),
                    "semantic_score": float(row["best_semantic_score"]),
                    "status": row["status"],
                    "polarity": row["polarity"],
                    "target_kind": row["target_kind"],
                }
            )

        out.sort(key=lambda r: (r["score"], r["memory"].memory_id), reverse=True)
        return out

    def _dedup_rows(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        out = []
        seen = set()
        for row in rows:
            mem_id = row["memory"].memory_id
            if mem_id in seen:
                continue
            out.append(row)
            seen.add(mem_id)
        return out

    def _select_diverse_slate(
        self,
        rows: list[dict[str, Any]],
        budget_k: int,
    ) -> list[dict[str, Any]]:
        selected: list[dict[str, Any]] = []
        selected_ids: set[str] = set()

        for row in rows:
            memory = row["memory"]
            mem_id = memory.memory_id
            if mem_id in selected_ids:
                continue

            redundant = False
            new_tokens = self._memory_tokens(self._retrieval_text(memory))
            for prev in selected:
                prev_tokens = self._memory_tokens(self._retrieval_text(prev["memory"]))
                if not new_tokens or not prev_tokens:
                    continue
                jacc = len(new_tokens & prev_tokens) / max(len(new_tokens | prev_tokens), 1)
                if jacc >= 0.80:
                    redundant = True
                    break

            if redundant:
                continue

            selected.append(row)
            selected_ids.add(mem_id)

            if len(selected) >= budget_k:
                break

        return selected

    def _retrieve_top_k(
        self,
        episode: dict[str, Any],
        active_memories: list[MemoryRecord],
        superseded_memories: list[MemoryRecord],
        k: int,
    ) -> tuple[list[MemoryRecord], list[float], list[dict[str, Any]]]:
        if not active_memories and not superseded_memories:
            return [], [], []

        question = episode.get("question", "") or ""
        options = episode.get("options", "") or ""
        query = f"{question} {options}"
        question_type = episode.get("question_type", "")
        focus_mode = self._query_focus_mode(episode)

        query_slot = self._infer_query_slot(question=question, options=options)

        filtered_active = active_memories
        filtered_superseded = superseded_memories

        if query_slot is not None and focus_mode not in {"raw_recall"}:
            slot_active = [m for m in active_memories if m.metadata.get("slot") == query_slot]
            slot_superseded = [m for m in superseded_memories if m.metadata.get("slot") == query_slot]

            if focus_mode == "slot_pref":
                if slot_active:
                    filtered_active = slot_active
                if slot_superseded:
                    filtered_superseded = slot_superseded
            else:
                if len(slot_active) >= 3:
                    filtered_active = slot_active
                if len(slot_superseded) >= 2:
                    filtered_superseded = slot_superseded

        effective_k = self.top_k
        if question_type in EVOLUTION_QUESTION_TYPES:
            effective_k = max(self.top_k, 7)

        dense_active_rows = self._scored_candidates(
            filtered_active,
            query=query,
            cache_key_prefix=f"controlled_dense_active_{episode.get('question_id', episode.get('episode_id'))}",
            episode=episode,
        )
        dense_active_rows = sorted(dense_active_rows, key=lambda r: (r["score"], r["memory"].memory_id), reverse=True)

        if self.ablation == "no_bm25":
            bm25_active_rows = []
        else:
            bm25_active_records = self._bm25_retrieve(filtered_active, query=query, k=20)
            bm25_active_rows = self._scored_candidates(
                bm25_active_records,
                query=query,
                cache_key_prefix=f"controlled_bm25_active_{episode.get('question_id', episode.get('episode_id'))}",
                episode=episode,
            )
            bm25_active_rows = sorted(bm25_active_rows, key=lambda r: (r["score"], r["memory"].memory_id), reverse=True)

        if self.ablation == "no_options":
            option_rows = []
        else:
            option_rows = self._option_conditioned_rows(
                filtered_active,
                question=question,
                options=options,
                episode=episode,
                per_option_k=4,
            )
            option_rows = sorted(option_rows, key=lambda r: (r["score"], r["memory"].memory_id), reverse=True)

        temporal_pool = filtered_active + filtered_superseded if focus_mode == "timeline" else filtered_active
        temporal_rows = self._scored_candidates(
            temporal_pool,
            query=query,
            cache_key_prefix=f"controlled_temporal_{episode.get('question_id', episode.get('episode_id'))}",
            episode=episode,
        )
        temporal_rows = sorted(
            temporal_rows,
            key=lambda r: (self._temporal_score(r["memory"], episode), r["score"], r["memory"].memory_id),
            reverse=True,
        )

        if focus_mode == "timeline":
            ranked_lists = [
                dense_active_rows[:20],
                bm25_active_rows[:20],
                option_rows[:20],
                temporal_rows[:20],
            ]
            fused_rows = self._rrf_fuse(ranked_lists)
            fused_rows = self._dedup_rows(fused_rows)

            pos_rows = [r for r in fused_rows if r["polarity"] == "positive"]
            neg_rows = [r for r in fused_rows if r["polarity"] == "negative"]
            sup_rows = [r for r in fused_rows if "superseded" in str(r["memory"].status).lower()]

            chosen_rows: list[dict[str, Any]] = []
            seen_ids: set[str] = set()

            for bucket in [pos_rows, neg_rows, sup_rows]:
                if bucket:
                    row = bucket[0]
                    mem_id = row["memory"].memory_id
                    if mem_id not in seen_ids:
                        chosen_rows.append(row)
                        seen_ids.add(mem_id)

            for row in fused_rows:
                mem_id = row["memory"].memory_id
                if mem_id in seen_ids:
                    continue
                chosen_rows.append(row)
                seen_ids.add(mem_id)
                if len(chosen_rows) >= max(effective_k, 7):
                    break

            chosen_rows = self._select_diverse_slate(chosen_rows, effective_k)
            all_debug_rows = fused_rows

        else:
            if focus_mode == "raw_recall":
                ranked_lists = [
                    dense_active_rows[:25],
                    bm25_active_rows[:25],
                    option_rows[:25],
                ]
            else:
                ranked_lists = [
                    dense_active_rows[:20],
                    bm25_active_rows[:20],
                    option_rows[:20],
                    temporal_rows[:20],
                ]

            fused_rows = self._rrf_fuse(ranked_lists)
            fused_rows = self._dedup_rows(fused_rows)
            chosen_rows = self._select_diverse_slate(fused_rows, effective_k)
            all_debug_rows = fused_rows

        if len(chosen_rows) < min(5, effective_k):
            fallback_rows = self._scored_candidates(
                active_memories,
                query=query,
                cache_key_prefix=f"controlled_fallback_{episode.get('question_id', episode.get('episode_id'))}",
                episode=episode,
            )
            fallback_rows = sorted(fallback_rows, key=lambda r: (r["score"], r["memory"].memory_id), reverse=True)

            merged = chosen_rows[:]
            seen = {row["memory"].memory_id for row in merged}

            for row in fallback_rows:
                mem_id = row["memory"].memory_id
                if mem_id in seen:
                    continue
                merged.append(row)
                seen.add(mem_id)
                if len(merged) >= effective_k:
                    break

            chosen_rows = self._select_diverse_slate(merged, effective_k)

        retrieved_records = [row["memory"] for row in chosen_rows]
        retrieved_scores = [float(row["score"]) for row in chosen_rows]

        retrieval_debug = []
        for row in all_debug_rows:
            memory = row["memory"]
            retrieval_debug.append(
                {
                    "memory_id": memory.memory_id,
                    "text": memory.normalized_mem_text,
                    "retrieval_text": self._retrieval_text(memory),
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
                status = str(memory.status)

                block = [f"- Memory: {canonical}", f"  Status: {status}"]
                if provenance:
                    block.append(f"  Evidence: {provenance}")

                memory_lines.append("\n".join(block))

            memories_block = "\n".join(memory_lines)
        else:
            memories_block = "None"

        question = episode.get("question", "") or ""
        options = episode.get("options", "") or ""

        return (
            "You are answering a multiple-choice question about a user.\n\n"
            "Use the retrieved memories and evidence carefully.\n"
            "Prefer ACTIVE memories as the user's current state.\n"
            "Use older or superseded evidence only when the question asks about change over time, "
            "reasons for updates, or preference evolution.\n\n"
            f"Retrieved memories:\n{memories_block}\n\n"
            f"Question: {question}\n"
            f"Options:\n{options}\n\n"
            "Return EXACTLY one of: (a) (b) (c) (d). No other text."
        )
