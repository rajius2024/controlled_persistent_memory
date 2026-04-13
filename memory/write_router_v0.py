from __future__ import annotations

import hashlib
import json
import os
import re
import time
from functools import lru_cache
from typing import Any

import numpy as np
from groq import Groq

from baselines.embeddings import embed_one
from memory.schema import MemoryRecord, MemoryStatus


WRITE_GATE_PROMPT = """
Extract durable user-specific memories from one user utterance.

Keep only stable facts, preferences, habits, or profile info useful later.
Reject temporary moods, one-off events, vague filler, and long explanations.

Return ONLY JSON:
{"items":[
  {"canonical_memory":"short normalized phrase",
   "slot":"food|drink|music|books|communication|schedule|location|profile|general",
   "target_kind":"topic|routine|social_style|profile",
   "polarity":"positive|negative|neutral|interest",
   "update_hint":false}
]}

Return at most 1 item unless the utterance clearly contains 2 distinct durable memories.

Set update_hint=true ONLY when the utterance explicitly signals a change from a prior state.

Required signals include phrases like:
- "used to"
- "no longer"
- "lost interest"
- "not anymore"
- "stopped"
- "moved away from"
- "gave up"
- "realized I no longer"

Do NOT set update_hint=true for:
- new information stated for the first time
- preferences or values introduced without contrast to a prior state
- general reflections, observations, or opinions
- broad self-description without an explicit change signal

Extract ALL distinct durable preference or profile signals from the utterance.
A single utterance may contain multiple memories.
Do not merge different preferences into one item.
Commonly missed: a negative update and a positive anchor in the same utterance.

Examples:
"likes producing music"
"dislikes mushrooms"
"prefers texting over calling"
"usually works late"
"is vegetarian"
"is from Seattle"

If nothing qualifies, return {"items":[]}.
"""


VALID_SLOTS = {
    "food",
    "drink",
    "music",
    "books",
    "communication",
    "schedule",
    "location",
    "profile",
    "general",
}

VALID_TARGET_KINDS = {
    "topic",
    "routine",
    "social_style",
    "profile",
}

VALID_MEMORY_TYPES = {
    "food_preference",
    "drink_preference",
    "music_preference",
    "book_preference",
    "communication_preference",
    "schedule_preference",
    "profile_fact",
    "habit_or_routine",
    "general_preference",
}

VALID_POLARITIES = {
    "positive",
    "negative",
    "neutral",
    "interest",
}


def _make_memory_id(persona_id: str, turn_index: int, normalized_text: str) -> str:
    raw = f"{persona_id}|{turn_index}|{normalized_text.strip().lower()}"
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]
    return f"mem_{digest}"


def _clean_text(text: str) -> str:
    text = str(text).strip()
    text = re.sub(r"\s+", " ", text)
    return text


def _truncate_clause(text: str) -> str:
    text = _clean_text(text)
    text = re.split(
        r"\b(?:because|since|especially|when|which|that|while|although|though)\b",
        text,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0]
    text = re.split(r"[.;:!?]", text, maxsplit=1)[0]
    return _clean_text(text)


def _canonicalize_prefix(text: str) -> str:
    text = text.strip().lower()

    replacements = [
        ("enjoys ", "likes "),
        ("enjoy ", "likes "),
        ("loves ", "likes "),
        ("love ", "likes "),
        ("hates ", "dislikes "),
        ("hate ", "dislikes "),
        ("never drinks ", "dislikes "),
        ("never eats ", "dislikes "),
        ("lives in ", "is from "),
        ("from ", "is from "),
        ("works best ", "prefers "),
        ("works better ", "prefers "),
    ]

    for old, new in replacements:
        if text.startswith(old):
            return new + text[len(old):]

    return text


def _normalize_canonical_memory(text: str) -> str:
    text = _truncate_clause(text).lower()
    text = re.sub(r"\s+", " ", text).strip()
    text = text.strip(" .,!?:;\"'")
    text = _canonicalize_prefix(text)
    return text


def _is_valid_canonical_memory(text: str) -> bool:
    if not text or len(text) < 5:
        return False

    toks = text.split()
    if len(toks) > 12:
        return False

    banned = {"because", "since", "which", "that", "while", "although", "though", "when"}
    if any(tok in banned for tok in toks):
        return False

    return True


def should_store_turn(turn: dict[str, Any]) -> bool:
    if turn.get("role") != "user":
        return False

    content = str(turn.get("content", "")).strip().lower()
    if len(content) <= 20:
        return False

    padded = f" {content} "

    first_person_markers = [
        " i ",
        " i'm ",
        " i’m ",
        " i am ",
        " i was ",
        " i've ",
        " i’ve ",
        " i'd ",
        " i’d ",
        " my ",
        " me ",
        " mine ",
    ]
    if not any(marker in padded for marker in first_person_markers):
        return False

    durable_markers = [
        "prefer",
        "like",
        "love",
        "enjoy",
        "hate",
        "dislike",
        "favorite",
        "usually",
        "always",
        "never",
        "used to",
        "from now on",
        "no longer",
        "interested in",
        "passionate about",
        "curious about",
        "rather than",
        "better when",
        "work better",
        "works better",
        "i'm from",
        "i am from",
        "i live in",
        "vegetarian",
        "vegan",
    ]
    if not any(marker in content for marker in durable_markers):
        return False

    transient_markers = [
        "today",
        "tonight",
        "yesterday",
        "last night",
        "this week",
        "for now",
    ]
    if any(marker in content for marker in transient_markers):
        return False

    return True


def _default_memory_type(slot: str, target_kind: str) -> str:
    slot_map = {
        "music": "music_preference",
        "books": "book_preference",
        "drink": "drink_preference",
        "food": "food_preference",
        "communication": "communication_preference",
        "schedule": "schedule_preference",
        "profile": "profile_fact",
    }
    if slot in slot_map:
        return slot_map[slot]
    if target_kind == "routine":
        return "habit_or_routine"
    return "general_preference"


@lru_cache(maxsize=32)
def _slot_anchor_vec(slot: str) -> tuple[float, ...]:
    vec = embed_one(f"{slot} in general")
    return tuple(float(x) for x in vec)


def compute_specificity_score(canonical_text: str, slot: str) -> float:
    mem_vec = np.array(embed_one(canonical_text), dtype=float)
    anchor_vec = np.array(_slot_anchor_vec(slot), dtype=float)

    mem_norm = np.linalg.norm(mem_vec)
    anchor_norm = np.linalg.norm(anchor_vec)
    if mem_norm == 0.0 or anchor_norm == 0.0:
        return 0.0

    sim = float(np.dot(mem_vec, anchor_vec) / (mem_norm * anchor_norm))
    sim = max(-1.0, min(1.0, sim))
    return 1.0 - sim


class GroqWriteGate:
    def __init__(
        self,
        model_name: str | None = None,
        max_completion_tokens: int = 64,
        api_key: str | None = None,
    ):
        self.model_name = model_name or os.environ.get(
            "WRITE_GATE_MODEL",
            "llama-3.3-70b-versatile",
        )
        print(f"[write_gate] loading model: {self.model_name}", flush=True)

        resolved_key = api_key or os.environ.get("GROQ_API_KEY")
        if not resolved_key:
            raise ValueError("GROQ_API_KEY is not set.")

        self.client = Groq(api_key=resolved_key)
        self.max_completion_tokens = max_completion_tokens
        self.cache: dict[str, list[dict[str, Any]]] = {}

    def _extract_json_text(self, text: str) -> str:
        text = text.strip()
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError(f"Model output did not contain JSON object: {text}")
        return text[start : end + 1]

    def extract_candidates(self, utterance: str) -> list[dict[str, Any]]:
        cache_key = f"{self.model_name}||{self.max_completion_tokens}||{utterance.strip()}"
        if cache_key in self.cache:
            return self.cache[cache_key]

        completion = self.client.chat.completions.create(
            model=self.model_name,
            temperature=0,
            response_format={"type": "json_object"},
            max_completion_tokens=self.max_completion_tokens,
            messages=[
                {"role": "system", "content": WRITE_GATE_PROMPT},
                {"role": "user", "content": utterance},
            ],
        )

        raw = completion.choices[0].message.content.strip()
        print(f"[write_gate][raw] {raw}", flush=True)

        try:
            raw_json = self._extract_json_text(raw)
            parsed = json.loads(raw_json)
        except Exception:
            self.cache[cache_key] = []
            return []

        items = parsed.get("items", [])
        if not isinstance(items, list):
            self.cache[cache_key] = []
            return []

        cleaned: list[dict[str, Any]] = []

        for item in items:
            canonical_memory = _normalize_canonical_memory(str(item.get("canonical_memory", "")))
            if not _is_valid_canonical_memory(canonical_memory):
                print(
                    f"[write_gate][drop] invalid_canonical raw={item.get('canonical_memory', '')!r} normalized={canonical_memory!r}",
                    flush=True,
                )
                continue

            slot = str(item.get("slot", "general")).strip().lower()
            if slot not in VALID_SLOTS:
                slot = "general"

            target_kind = str(item.get("target_kind", "topic")).strip().lower()
            if target_kind not in VALID_TARGET_KINDS:
                target_kind = "topic"

            polarity = str(item.get("polarity", "neutral")).strip().lower()
            if polarity not in VALID_POLARITIES:
                polarity = "neutral"

            memory_type = _default_memory_type(slot, target_kind)

            cleaned.append(
                {
                    "normalized_text": canonical_memory,
                    "slot": slot,
                    "target_kind": target_kind,
                    "polarity": polarity,
                    "memory_type": memory_type,
                    "extraction_reason": "groq_write_gate",
                    "update_hint": bool(item.get("update_hint", False)),
                    "confidence": 1.0,
                }
            )

        self.cache[cache_key] = cleaned
        return cleaned


_GATE: GroqWriteGate | None = None


def _get_gate() -> GroqWriteGate:
    global _GATE
    if _GATE is None:
        _GATE = GroqWriteGate()
    return _GATE


def _build_memory_record(
    persona_id: str,
    turn_index: int,
    source_text: str,
    candidate: dict[str, Any],
) -> MemoryRecord:
    normalized = candidate["normalized_text"]
    memory_id = _make_memory_id(persona_id, turn_index, normalized)

    return MemoryRecord(
        memory_id=memory_id,
        persona_id=persona_id,
        turn_index=turn_index,
        status=MemoryStatus.ACTIVE,
        normalized_mem_text=normalized,
        memory_type=candidate["memory_type"],
        metadata={
            "source_utterance": source_text,
            "provenance_text": source_text[:300],
            "router_version": "write_gate_v5_groq_specificity",
            "source_turn_index": turn_index,
            "slot": candidate["slot"],
            "polarity": candidate["polarity"],
            "target_kind": candidate["target_kind"],
            "confidence": candidate.get("confidence", 1.0),
            "extraction_reason": candidate["extraction_reason"],
            "update_hint": candidate["update_hint"],
            "specificity_score": compute_specificity_score(
                normalized,
                candidate["slot"],
            ),
        },
    )


def extract_memories_from_turn(turn: dict[str, Any], persona_id: str) -> list[MemoryRecord]:
    if not should_store_turn(turn):
        return []

    content = _clean_text(turn.get("content", ""))
    content = content[:450]
    if not content:
        return []

    turn_index = int(turn.get("turn_index", 0))
    gate = _get_gate()

    start = time.perf_counter()
    candidates = gate.extract_candidates(content)
    elapsed = time.perf_counter() - start
    print(f"[write_gate] persona={persona_id} turn={turn_index} took {elapsed:.2f}s", flush=True)

    records: list[MemoryRecord] = []
    seen: set[tuple[str, str, str, str]] = set()

    for candidate in candidates:
        key = (
            candidate["normalized_text"],
            candidate["slot"],
            candidate["target_kind"],
            candidate["polarity"],
        )
        if key in seen:
            continue
        seen.add(key)
        records.append(_build_memory_record(persona_id, turn_index, content, candidate))

    return records


def extract_memories_from_episode(episode: dict[str, Any]) -> list[MemoryRecord]:
    persona_id = str(episode["persona_id"])
    turns = episode.get("turns", [])

    memories: list[MemoryRecord] = []
    for turn in turns:
        memories.extend(extract_memories_from_turn(turn, persona_id))
    return memories
