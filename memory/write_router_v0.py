from __future__ import annotations

import hashlib
import re
from typing import Any

from memory.schema import MemoryRecord, MemoryStatus


def _make_memory_id(persona_id: str, turn_index: int, normalized_text: str) -> str:
    raw = f"{persona_id}|{turn_index}|{normalized_text.strip().lower()}"
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]
    return f"mem_{digest}"


def _classify_memory_type(text: str) -> str:
    lowered = text.lower()

    if any(token in lowered for token in ["coffee", "tea", "juice", "drink", "soda", "music", "song", "playlist", "genre", "beats"]):
        return "music_preference" if any(token in lowered for token in ["music", "song", "playlist", "genre", "beats"]) else "drink_preference"

    if any(token in lowered for token in ["pizza", "pasta", "spicy", "food", "breakfast", "dinner", "cuisine"]):
        return "food_preference"

    if any(token in lowered for token in ["text", "phone", "call", "email"]):
        return "communication_preference"

    if any(token in lowered for token in ["morning", "night", "weekend", "weekday"]):
        return "schedule_preference"

    if "color" in lowered:
        return "color_preference"

    return "general_preference"


def _clean_tail(value: str) -> str:
    value = value.strip()
    value = re.sub(r"[.?!]+$", "", value).strip()
    value = re.sub(r"\s+", " ", value)
    return value


def _normalize_text(raw_text: str) -> str | None:
    text = raw_text.strip()

    favorite_match = re.search(
        r"\bmy\s+favorite\s+(?P<slot>\w+)\s+is\s+(?P<value>.+)",
        text,
        re.IGNORECASE,
    )
    if favorite_match:
        slot = favorite_match.group("slot").strip().lower()
        value = _clean_tail(favorite_match.group("value"))
        return f"favorite {slot} is {value}"

    negative_match = re.search(
        r"\bI\s+(?:do\s+not|don't|dont)\s+like\s+(?P<value>.+)",
        text,
        re.IGNORECASE,
    )
    if negative_match:
        value = _clean_tail(negative_match.group("value"))
        return f"dislikes {value}"

    prefer_match = re.search(
        r"\bI\s+prefer\s+(?P<value>.+)",
        text,
        re.IGNORECASE,
    )
    if prefer_match:
        value = _clean_tail(prefer_match.group("value"))
        return f"prefers {value}"

    like_match = re.search(
        r"\bI\s+like\s+(?P<value>.+)",
        text,
        re.IGNORECASE,
    )
    if like_match:
        value = _clean_tail(like_match.group("value"))
        return f"likes {value}"

    love_match = re.search(
        r"\bI\s+love\s+(?P<value>.+)",
        text,
        re.IGNORECASE,
    )
    if love_match:
        value = _clean_tail(love_match.group("value"))
        return f"likes {value}"

    enjoy_match = re.search(
        r"\bI\s+enjoy\s+(?P<value>.+)",
        text,
        re.IGNORECASE,
    )
    if enjoy_match:
        value = _clean_tail(enjoy_match.group("value"))
        return f"enjoys {value}"

    interested_match = re.search(
        r"\bI\s+(?:am\s+|\'m\s+)?interested\s+in\s+(?P<value>.+)",
        text,
        re.IGNORECASE,
    )
    if interested_match:
        value = _clean_tail(interested_match.group("value"))
        return f"interested in {value}"

    from_now_on_match = re.search(
        r"\bfrom\s+now\s+on\b[,:\s]*(?P<value>.+)",
        text,
        re.IGNORECASE,
    )
    if from_now_on_match:
        value = _clean_tail(from_now_on_match.group("value"))
        return value.lower()

    return None


def should_store_turn(turn: dict[str, Any]) -> bool:
    if turn.get("role") != "user":
        return False

    content = str(turn.get("content", ""))
    lowered = content.lower()

    trigger_terms = [
        "prefer",
        "like",
        "don't",
        "dont",
        "from now on",
        "favorite",
        "enjoy",
        "interested in",
        "interested",
    ]
    return any(term in lowered for term in trigger_terms)


def extract_memories_from_turn(
    turn: dict[str, Any],
    persona_id: str,
) -> list[MemoryRecord]:
    if not should_store_turn(turn):
        return []

    content = str(turn.get("content", "")).strip()
    turn_index = int(turn.get("turn_index", 0))

    normalized = _normalize_text(content)
    if not normalized:
        return []
    if not _is_memory_worthy(normalized):
        return []

    memory_type = _classify_memory_type(normalized)
    memory_id = _make_memory_id(persona_id, turn_index, normalized)

    record = MemoryRecord(
        memory_id=memory_id,
        persona_id=persona_id,
        turn_index=turn_index,
        status=MemoryStatus.ACTIVE,
        normalized_mem_text=normalized,
        memory_type=memory_type,
        metadata={
            "source_utterance": content,
            "router_version": "write_router_v0_1",
            "source_turn_index": turn_index,
            "extraction_note": "rule_based_trigger_match",
        },
    )
    return [record]


def extract_memories_from_episode(episode: dict[str, Any]) -> list[MemoryRecord]:
    persona_id = str(episode["persona_id"])
    turns = episode.get("turns", [])

    memories: list[MemoryRecord] = []
    for turn in turns:
        memories.extend(extract_memories_from_turn(turn, persona_id))
    return memories

def _is_memory_worthy(normalized_text: str) -> bool:
    lowered = normalized_text.strip().lower()

    reject_prefixes = [
        "enjoys myself",
        "enjoys it",
        "enjoys this",
        "enjoys that",
        "enjoys being",
        "enjoys meeting",
        "enjoys talking",
        "enjoys spending time",
    ]

    if any(lowered.startswith(prefix) for prefix in reject_prefixes):
        return False

    return True
