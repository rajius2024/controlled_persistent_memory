from __future__ import annotations

import math
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

NEGATIVE_MARKERS = (
    "dislikes",
    "doesn't like",
    "does not like",
    "discontinued",
    "stopped",
    "gave up",
    "canceled",
    "cancelled",
    "avoids",
    "avoid",
    "not interested",
    "overwhelming",
    "stressful",
    "arguments",
    "conflict",
    "frustrating",
    "emotionally draining",
    "not worth",
)

POSITIVE_MARKERS = (
    "likes",
    "prefers",
    "enjoys",
    "values",
    "appreciates",
    "loves",
    "has",
    "started",
    "interested in",
    "wants",
)

CAUSE_MARKERS = (
    "because",
    "as i realized",
    "i realized",
    "led to",
    "due to",
    "since",
    "made me",
    "became",
    "turned into",
    "resulted in",
    "overwhelming",
    "stressful",
    "arguments",
    "conflict",
    "emotionally draining",
    "frustrating",
)

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
            "instruments", "rhythm", "melody", "soundtrack", "film score",
        ]):
            return "music"

        if any(token in combined for token in [
            "book", "books", "reading", "read", "reader", "novel", "novels",
            "literature", "literary", "author", "authors", "bookstore",
            "bookstores", "library", "libraries", "blog", "blogging",
            "fiction", "nonfiction", "poetry",
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

    def _target_tokens(self, text: str) -> set[str]:
        lowered = text.strip().lower()
        prefixes = [
            "likes ",
            "dislikes ",
            "prefers ",
            "usually ",
            "is ",
            "interested in ",
        ]
        for prefix in prefixes:
            if lowered.startswith(prefix):
                lowered = lowered[len(prefix):].strip()
                break
        return {token for token in re.findall(r"[a-z0-9]+", lowered) if token}

    def _retrieval_text(self, memory: MemoryRecord) -> str:
        canonical = memory.normalized_mem_text
        provenance = str(memory.metadata.get("provenance_text", "")).strip()
        status = str(memory.status)
        if provenance:
            return f"{canonical}. Status: {status}. Evidence: {provenance}"
        return f"{canonical}. Status: {status}"

    def _query_focus_mode(self, episode: dict[str, Any]) -> str:
        qtype = episode.get("question_type", "") or ""

        if qtype in {"recall_user_shared_facts", "recalling_facts_mentioned_by_the_user"}:
            return "raw_recall"

        if qtype == "recalling_the_reasons_behind_previous_updates":
            return "reason_update"

        if qtype == "track_full_preference_evolution":
            return "timeline"

        if qtype in {
            "bookRecommendation",
            "musicRecommendation",
            "movieRecommendation",
            "provide_preference_aligned_recommendations",
        }:
            return "recommendation"

        if qtype == "suggest_new_ideas":
            return "idea_generation"

        return "default"

    def _specificity_bonus(self, memory: MemoryRecord) -> float:
        text = memory.normalized_mem_text.lower()
        tokens = self._memory_tokens(text)

        bonus = 0.0
        bonus += min(len(tokens), 8) * 0.015

        if any(
            word in text
            for word in ["producing", "creates", "started", "hosts", "joined", "volunteers", "visits", "uses"]
        ):
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

    def _shadowed_generic_penalty(
        self,
        memory: MemoryRecord,
        pool: list[MemoryRecord],
    ) -> float:
        mem_tokens = self._target_tokens(memory.normalized_mem_text)
        if not mem_tokens:
            return 0.0

        penalty = 0.0
        slot = memory.metadata.get("slot")
        target_kind = memory.metadata.get("target_kind")
        polarity = memory.metadata.get("polarity")

        for other in pool:
            if other.memory_id == memory.memory_id:
                continue
            if other.metadata.get("slot") != slot:
                continue
            if other.metadata.get("target_kind") != target_kind:
                continue
            if other.metadata.get("polarity") != polarity:
                continue

            other_tokens = self._target_tokens(other.normalized_mem_text)
            if not other_tokens:
                continue

            if mem_tokens.issubset(other_tokens) and len(mem_tokens) < len(other_tokens):
                penalty = max(penalty, 0.18)

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
        return 0.18 * overlap

    def _literal_overlap_bonus(self, memory: MemoryRecord, query: str) -> float:
        mem_tokens = self._memory_tokens(self._retrieval_text(memory))
        query_tokens = self._memory_tokens(query)
        if not mem_tokens or not query_tokens:
            return 0.0
        overlap = len(mem_tokens & query_tokens) / max(len(query_tokens), 1)
        return 0.20 * overlap

    def _reason_signal_bonus(self, memory: MemoryRecord) -> float:
        provenance = str(memory.metadata.get("provenance_text", "")).lower()
        text = memory.normalized_mem_text.lower()

        score = 0.0
        reason_markers = [
            "because",
            "led to",
            "so that",
            "it became clear",
            "i realized",
            "frustrating",
            "stressful",
            "overwhelming",
            "contentious",
            "emotionally draining",
            "conflict",
            "arguments",
        ]
        if any(marker in provenance for marker in reason_markers):
            score += 0.12

        if any(marker in text for marker in ["stopped", "discontinued", "gave up", "no longer"]):
            score += 0.06

        return score

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

    def _rerank_score(
        self,
        semantic_score: float,
        memory: MemoryRecord,
        query: str,
        episode: dict[str, Any],
        candidate_pool: list[MemoryRecord],
    ) -> float:
        specificity = float(memory.metadata.get("specificity_score", 0.0))
        return (
            semantic_score
            + self._specificity_bonus(memory)
            - self._generic_penalty(memory)
            - self._shadowed_generic_penalty(memory, candidate_pool)
            + 0.15 * specificity
            + self._evidence_bonus(memory, query)
            + self._temporal_score(memory, episode)
        )

    def _mode_adjusted_score(
        self,
        row: dict[str, Any],
        query: str,
        focus_mode: str,
    ) -> float:
        memory = row["memory"]
        score = float(row["score"])

        if focus_mode == "raw_recall":
            score += self._literal_overlap_bonus(memory, query)
            score += self._evidence_bonus(memory, query) * 0.75

        elif focus_mode == "reason_update":
            score += self._reason_signal_bonus(memory)
            if "superseded" in str(memory.status).lower():
                score += 0.08
            if bool(memory.metadata.get("update_hint", False)):
                score += 0.08

        elif focus_mode == "recommendation":
            slot = memory.metadata.get("slot")
            if slot in {"music", "books", "food", "drink"}:
                score += 0.04

        elif focus_mode == "idea_generation":
            provenance = str(memory.metadata.get("provenance_text", "")).lower()
            if any(tok in provenance for tok in ["started", "planned", "introduced", "created", "challenge", "tradition"]):
                score += 0.06

        return score

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
            final = self._rerank_score(
                semantic,
                memory,
                query,
                episode,
                memories,
            )
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

    def _build_mode_structured_rows(
        self,
        fused_rows: list[dict[str, Any]],
        focus_mode: str,
        effective_k: int,
    ) -> list[dict[str, Any]]:
        chosen_rows: list[dict[str, Any]] = []
        seen_ids: set[str] = set()

        def add_first(rows: list[dict[str, Any]]) -> None:
            for row in rows:
                mem_id = row["memory"].memory_id
                if mem_id not in seen_ids:
                    chosen_rows.append(row)
                    seen_ids.add(mem_id)
                    return

        if focus_mode == "timeline":
            active_rows = [r for r in fused_rows if "active" in str(r["memory"].status).lower()]
            changed_rows = [
                r for r in fused_rows
                if "superseded" in str(r["memory"].status).lower()
                or bool(r["memory"].metadata.get("update_hint", False))
            ]
            pos_rows = [r for r in fused_rows if r["polarity"] == "positive"]
            neg_rows = [r for r in fused_rows if r["polarity"] == "negative"]

            for bucket in [active_rows, changed_rows, pos_rows, neg_rows]:
                add_first(bucket)

        elif focus_mode == "reason_update":
            active_rows = [r for r in fused_rows if "active" in str(r["memory"].status).lower()]
            changed_rows = [
                r for r in fused_rows
                if "superseded" in str(r["memory"].status).lower()
                or bool(r["memory"].metadata.get("update_hint", False))
            ]
            reason_rows = sorted(
                fused_rows,
                key=lambda r: self._reason_signal_bonus(r["memory"]),
                reverse=True,
            )

            for bucket in [changed_rows, reason_rows, active_rows]:
                add_first(bucket)

        for row in fused_rows:
            mem_id = row["memory"].memory_id
            if mem_id in seen_ids:
                continue
            chosen_rows.append(row)
            seen_ids.add(mem_id)
            if len(chosen_rows) >= max(effective_k, 7 if focus_mode in {"timeline", "reason_update"} else effective_k):
                break

        return chosen_rows

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
        focus_mode = self._query_focus_mode(episode)

        query_slot = self._infer_query_slot(question=question, options=options)

        filtered_active = active_memories
        filtered_superseded = superseded_memories

        if query_slot is not None and focus_mode not in {"raw_recall"}:
            slot_active = [m for m in active_memories if m.metadata.get("slot") == query_slot]
            slot_superseded = [m for m in superseded_memories if m.metadata.get("slot") == query_slot]

            if focus_mode in {"recommendation"}:
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
        if focus_mode in {"timeline", "reason_update"}:
            effective_k = max(self.top_k, 7)

        dense_active_rows = self._scored_candidates(
            filtered_active,
            query=query,
            cache_key_prefix=f"controlled_dense_active_{episode.get('question_id', episode.get('episode_id'))}",
            episode=episode,
        )
        dense_active_rows = sorted(dense_active_rows, key=lambda r: (r["score"], r["memory"].memory_id), reverse=True)

        bm25_active_records = self._bm25_retrieve(filtered_active, query=query, k=20)
        bm25_active_rows = self._scored_candidates(
            bm25_active_records,
            query=query,
            cache_key_prefix=f"controlled_bm25_active_{episode.get('question_id', episode.get('episode_id'))}",
            episode=episode,
        )
        bm25_active_rows = sorted(bm25_active_rows, key=lambda r: (r["score"], r["memory"].memory_id), reverse=True)

        option_rows = self._option_conditioned_rows(
            filtered_active,
            question=question,
            options=options,
            episode=episode,
            per_option_k=5 if focus_mode in {"recommendation", "idea_generation"} else 4,
        )
        option_rows = sorted(option_rows, key=lambda r: (r["score"], r["memory"].memory_id), reverse=True)

        temporal_pool = filtered_active + filtered_superseded if focus_mode in {"timeline", "reason_update"} else filtered_active
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

        if focus_mode == "raw_recall":
            ranked_lists = [
                dense_active_rows[:25],
                bm25_active_rows[:30],
                option_rows[:20],
            ]
        elif focus_mode == "reason_update":
            ranked_lists = [
                dense_active_rows[:20],
                bm25_active_rows[:20],
                option_rows[:20],
                temporal_rows[:25],
            ]
        elif focus_mode == "timeline":
            ranked_lists = [
                dense_active_rows[:20],
                bm25_active_rows[:20],
                option_rows[:20],
                temporal_rows[:25],
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

        adjusted_rows = []
        for row in fused_rows:
            adjusted = dict(row)
            adjusted["score"] = self._mode_adjusted_score(row, query, focus_mode)
            adjusted_rows.append(adjusted)

        adjusted_rows = sorted(
            adjusted_rows,
            key=lambda r: (r["score"], r["memory"].memory_id),
            reverse=True,
        )

        chosen_rows = self._build_mode_structured_rows(
            adjusted_rows,
            focus_mode=focus_mode,
            effective_k=effective_k,
        )

        if focus_mode == "raw_recall":
            chosen_rows = chosen_rows[:effective_k]
        else:
            chosen_rows = self._select_diverse_slate(chosen_rows, effective_k)

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

            if focus_mode == "raw_recall":
                chosen_rows = merged[:effective_k]
            else:
                chosen_rows = self._select_diverse_slate(merged, effective_k)

        retrieved_records = [row["memory"] for row in chosen_rows]
        retrieved_scores = [float(row["score"]) for row in chosen_rows]

        retrieval_debug = []
        for row in adjusted_rows:
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
    
    def _is_negative_memory(self, memory: MemoryRecord) -> bool:
        text = f"{memory.normalized_mem_text} {memory.metadata.get('provenance_text', '')}".lower()
        return any(marker in text for marker in NEGATIVE_MARKERS)

    def _is_positive_memory(self, memory: MemoryRecord) -> bool:
        text = f"{memory.normalized_mem_text} {memory.metadata.get('provenance_text', '')}".lower()
        return any(marker in text for marker in POSITIVE_MARKERS)

    def _has_causal_signal(self, memory: MemoryRecord) -> bool:
        text = f"{memory.normalized_mem_text} {memory.metadata.get('provenance_text', '')}".lower()
        return any(marker in text for marker in CAUSE_MARKERS)

    def _build_prompt(self, episode: dict[str, Any], retrieved_records: list[MemoryRecord]) -> str:
            focus_mode = self._query_focus_mode(episode)

            if retrieved_records:
                memory_lines = []

                if focus_mode == "raw_recall":
                    ordered = sorted(
                        retrieved_records,
                        key=lambda m: (-len(str(m.metadata.get("provenance_text", "")).strip()), m.memory_id),
                )[:3]

                    memory_lines.append("Relevant factual memories:")
                    for memory in ordered:
                        canonical = memory.normalized_mem_text
                        provenance = str(memory.metadata.get("provenance_text", "")).strip()
                        status = str(memory.status)

                        block = []
                        if provenance:
                            block.append(f"- Evidence: {provenance}")
                        block.append(f"  Memory summary: {canonical}")
                        block.append(f"  Status: {status}")
                        memory_lines.append("\n".join(block))

                    memory_lines.append(
                        "\nInstruction: Answer from the most direct factual memory. "
                        "Do not overuse broad background memories."
                    )

                elif focus_mode == "timeline":
                    active_rows = []
                    older_rows = []
                    other_rows = []

                    for memory in retrieved_records:
                        status = str(memory.status).lower()
                        if "superseded" in status or bool(memory.metadata.get("update_hint", False)):
                            older_rows.append(memory)
                        elif "active" in status:
                            active_rows.append(memory)
                        else:
                            other_rows.append(memory)

                    memory_lines.append("Relevant memory timeline:")

                    if older_rows:
                        memory_lines.append("\nOlder or superseded memories:")
                        for memory in older_rows[:3]:
                            canonical = memory.normalized_mem_text
                            provenance = str(memory.metadata.get("provenance_text", "")).strip()
                            status = str(memory.status)

                            block = [f"- {canonical}", f"  Status: {status}"]
                            if provenance:
                                block.append(f"  Evidence: {provenance}")
                            memory_lines.append("\n".join(block))

                    if active_rows:
                        memory_lines.append("\nCurrent active memories:")
                        for memory in active_rows[:5]:
                            canonical = memory.normalized_mem_text
                            provenance = str(memory.metadata.get("provenance_text", "")).strip()
                            status = str(memory.status)

                            block = [f"- {canonical}", f"  Status: {status}"]
                            if provenance:
                                block.append(f"  Evidence: {provenance}")
                            memory_lines.append("\n".join(block))

                    if other_rows:
                        memory_lines.append("\nOther relevant memories:")
                        for memory in other_rows[:2]:
                            canonical = memory.normalized_mem_text
                            provenance = str(memory.metadata.get("provenance_text", "")).strip()
                            status = str(memory.status)

                            block = [f"- {canonical}", f"  Status: {status}"]
                            if provenance:
                                block.append(f"  Evidence: {provenance}")
                            memory_lines.append("\n".join(block))

                    memory_lines.append(
                        "\nInstruction: Use active memories as the user's current state. "
                        "Use older or superseded memories only to understand how the preference changed over time."
                    )

                elif focus_mode == "reason_update":
                    causal_rows = []
                    other_rows = []

                    for memory in retrieved_records:
                        if self._has_causal_signal(memory) or self._is_negative_memory(memory):
                            causal_rows.append(memory)
                        else:
                            other_rows.append(memory)

                    ordered = causal_rows[:6] + other_rows[:2]

                    memory_lines.append("Relevant causal evidence:")
                    for memory in ordered:
                        canonical = memory.normalized_mem_text
                        provenance = str(memory.metadata.get("provenance_text", "")).strip()
                        status = str(memory.status)

                        block = [f"- Memory summary: {canonical}", f"  Status: {status}"]
                        if provenance:
                            block.insert(0, f"- Evidence: {provenance}")
                        memory_lines.append("\n".join(block))

                    memory_lines.append(
                        "\nInstruction: Prioritize explicit reasons such as because, led to, stressful, conflict, "
                        "arguments, overwhelming, emotionally draining, or similar causal evidence."
                    )

                elif focus_mode in {"recommendation", "idea_generation"}:
                    likes = []
                    avoids = []
                    neutral = []

                    for memory in retrieved_records:
                        if self._is_negative_memory(memory):
                            avoids.append(memory)
                        elif self._is_positive_memory(memory):
                            likes.append(memory)
                        else:
                            neutral.append(memory)

                    memory_lines.append("Relevant preference evidence:")

                    if likes:
                        memory_lines.append("\nLikes / preferences to support:")
                        for memory in likes[:5]:
                            canonical = memory.normalized_mem_text
                            provenance = str(memory.metadata.get("provenance_text", "")).strip()
                            status = str(memory.status)

                            block = [f"- {canonical}", f"  Status: {status}"]
                            if provenance:
                                block.append(f"  Evidence: {provenance}")
                            memory_lines.append("\n".join(block))

                    if avoids:
                        memory_lines.append("\nAvoid / dislike constraints:")
                        for memory in avoids[:4]:
                            canonical = memory.normalized_mem_text
                            provenance = str(memory.metadata.get("provenance_text", "")).strip()
                            status = str(memory.status)

                            block = [f"- {canonical}", f"  Status: {status}"]
                            if provenance:
                                block.append(f"  Evidence: {provenance}")
                            memory_lines.append("\n".join(block))

                    if neutral:
                        memory_lines.append("\nOther context:")
                        for memory in neutral[:2]:
                            canonical = memory.normalized_mem_text
                            provenance = str(memory.metadata.get("provenance_text", "")).strip()
                            status = str(memory.status)

                            block = [f"- {canonical}", f"  Status: {status}"]
                            if provenance:
                                block.append(f"  Evidence: {provenance}")
                            memory_lines.append("\n".join(block))

                    memory_lines.append(
                        "\nInstruction: Use likes/preferences as positive evidence. "
                        "Use avoid/dislike constraints only to eliminate bad options. "
                        "Do not choose an option similar to something the user disliked, stopped, gave up, "
                        "discontinued, canceled, or found stressful."
                    )

                else:
                    evidence_rich = [
                        memory for memory in retrieved_records
                        if str(memory.metadata.get("provenance_text", "")).strip()
                    ]
                    evidence_poor = [
                        memory for memory in retrieved_records
                        if not str(memory.metadata.get("provenance_text", "")).strip()
                    ]
                    ordered = evidence_rich + evidence_poor

                    memory_lines.append("Relevant memories:")
                    for memory in ordered[:5]:
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
                "The memory section may be organized by question type.\n\n"
                "For evolution questions, distinguish current active preferences from older or superseded preferences.\n"
                "For recommendation or suggestion questions, use likes/preferences as support and avoid/dislike memories as constraints.\n"
                "For reason questions, prioritize explicit causal evidence.\n"
                "For recall questions, prefer the most direct factual memory over broad background context.\n\n"
                f"Retrieved memories:\n{memories_block}\n\n"
                f"Question: {question}\n"
                f"Options:\n{options}\n\n"
                "Return EXACTLY one of: (a) (b) (c) (d). No other text."
            )
