from __future__ import annotations

import hashlib
import re
from typing import Any


PREFERENCE_PATTERNS = [
    ("favorite_pattern", re.compile(r"\bmy\s+favorite\s+(?P<slot>\w+)\s+is\s+(?P<value>.+)", re.IGNORECASE)),
    ("negative_preference_pattern", re.compile(r"\bI\s+(?:do\s+not|don't|dont)\s+like\s+(?P<value>.+)", re.IGNORECASE)),
    ("prefer_pattern", re.compile(r"\bI\s+prefer\s+(?P<value>.+)", re.IGNORECASE)),
    ("like_pattern", re.compile(r"\bI\s+like\s+(?P<value>.+)", re.IGNORECASE)),
    ("love_pattern", re.compile(r"\bI\s+love\s+(?P<value>.+)", re.IGNORECASE)),
    ("enjoy_pattern", re.compile(r"\bI\s+enjoy\s+(?P<value>.+)", re.IGNORECASE)),
    ("interest_pattern", re.compile(r"\bI\s+(?:am\s+|\'m\s+)?interested\s+in\s+(?P<value>.+)", re.IGNORECASE)),
    ("from_now_on_pattern", re.compile(r"\bfrom\s+now\s+on\b[,:\s]*(?P<value>.+)", re.IGNORECASE)),
]

TRIGGER_TERMS = [
    "prefer",
    "like",
    "love",
    "don't",
    "dont",
    "favorite",
    "from now on",
    "enjoy",
    "interested in",
    "interested",
]


def _make_memory_id(persona_id: str, turn_index: int, normalized_text: str) -> str:
    raw = f"{persona_id}|{turn_index}|{normalized_text.strip().lower()}"
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]
    return f"mem_{digest}"


def _clean_tail(value: str) -> str:
    value = value.strip()
    value = re.sub(r"[.?!]+$", "", value).strip()
    value = re.sub(r"\s+", " ", value)
    return value


def _trim_preference_value(value: str) -> str:
    value = value.strip()

    split_patterns = [
        r"\bbecause\b",
        r"\bsince\b",
        r"\bespecially\b",
        r"\bwhen\b",
        r"\bwhich\b",
        r"\bthat\b",
        r"\bas this\b",
        r"\brather than\b",
    ]

    for pattern in split_patterns:
        parts = re.split(pattern, value, maxsplit=1, flags=re.IGNORECASE)
        if len(parts) > 1:
            value = parts[0].strip()
            break

    value = re.split(r"[;,]", value, maxsplit=1)[0].strip()
    value = re.sub(r"\s+", " ", value)
    value = re.sub(r"[.?!]+$", "", value).strip()
    return value


def _infer_slot(text: str) -> str:
    lowered = text.lower()

    if any(token in lowered for token in ["music", "song", "playlist", "genre", "beats", "jazz", "rock", "pop"]):
        return "music"
    if any(token in lowered for token in ["coffee", "tea", "juice", "drink", "soda"]):
        return "drink"
    if any(token in lowered for token in ["pizza", "pasta", "spicy", "food", "breakfast", "dinner", "cuisine"]):
        return "food"
    if any(token in lowered for token in ["text", "phone", "call", "email"]):
        return "communication"
    if any(token in lowered for token in ["morning", "night", "weekend", "weekday"]):
        return "schedule"
    if "color" in lowered:
        return "color"

    return "general"


def _classify_memory_type(slot: str, text: str) -> str:
    if slot == "music":
        return "music_preference"
    if slot == "drink":
        return "drink_preference"
    if slot == "food":
        return "food_preference"
    if slot == "communication":
        return "communication_preference"
    if slot == "schedule":
        return "schedule_preference"
    if slot == "color":
        return "color_preference"

    lowered = text.lower()
    if any(token in lowered for token in ["usually", "always", "every morning", "every night", "tend to"]):
        return "habit_or_routine"

    return "general_preference"


def should_store_turn(turn: dict[str, Any]) -> bool:
    if turn.get("role") != "user":
        return False

    content = str(turn.get("content", ""))
    lowered = content.lower()
    return any(term in lowered for term in TRIGGER_TERMS)


def _extract_candidate(raw_text: str) -> dict[str, Any] | None:
    text = raw_text.strip()

    for reason, pattern in PREFERENCE_PATTERNS:
        match = pattern.search(text)
        if not match:
            continue

        if reason == "favorite_pattern":
            slot = match.group("slot").strip().lower()
            value = _clean_tail(_trim_preference_value(match.group("value")))
            normalized = f"favorite {slot} is {value}"
            return {
                "normalized_text": normalized,
                "polarity": "positive",
                "slot": slot,
                "extraction_reason": reason,
                "update_hint": False,
            }

        value = _clean_tail(_trim_preference_value(match.group("value")))
        if not value:
            return None

        if reason == "negative_preference_pattern":
            normalized = f"dislikes {value}"
            polarity = "negative"
        elif reason == "prefer_pattern":
            normalized = f"prefers {value}"
            polarity = "positive"
        elif reason in ("like_pattern", "love_pattern"):
            normalized = f"likes {value}"
            polarity = "positive"
        elif reason == "enjoy_pattern":
            normalized = f"enjoys {value}"
            polarity = "positive"
        elif reason == "interest_pattern":
            normalized = f"interested in {value}"
            polarity = "interest"
        elif reason == "from_now_on_pattern":
            normalized = value.lower()
            polarity = "positive"
        else:
            return None

        slot = _infer_slot(normalized)
        return {
            "normalized_text": normalized,
            "polarity": polarity,
            "slot": slot,
            "extraction_reason": reason,
            "update_hint": reason == "from_now_on_pattern" or any(
                marker in text.lower() for marker in ["from now on", "not anymore", "used to", "now prefer"]
            ),
        }

    return None


def _is_memory_worthy(candidate: dict[str, Any]) -> bool:
    lowered = candidate["normalized_text"].strip().lower()

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


def _compute_write_score(candidate: dict[str, Any], source_text: str) -> int:
    score = 0
    lowered_source = source_text.lower()

    if candidate["extraction_reason"] in {"favorite_pattern", "prefer_pattern", "negative_preference_pattern"}:
        score += 2
    else:
        score += 1

    if candidate["slot"] != "general":
        score += 1

    if any(marker in lowered_source for marker in ["always", "usually", "never", "favorite", "from now on"]):
        score += 1

    if candidate["update_hint"]:
        score += 1

    if len(candidate["normalized_text"].split()) > 20:
        score -= 1

    return score


def extract_memories_from_turn(turn: dict[str, Any], persona_id: str) -> list[dict[str, Any]]:
    if not should_store_turn(turn):
        return []

    content = str(turn.get("content", "")).strip()
    turn_index = int(turn.get("turn_index", 0))

    candidate = _extract_candidate(content)
    if not candidate:
        return []

    if not _is_memory_worthy(candidate):
        return []

    write_score = _compute_write_score(candidate, content)
    if write_score < 2:
        return []

    memory_type = _classify_memory_type(candidate["slot"], candidate["normalized_text"])
    memory_id = _make_memory_id(persona_id, turn_index, candidate["normalized_text"])

    return [{
        "memory_id": memory_id,
        "persona_id": persona_id,
        "turn_index": turn_index,
        "normalized_mem_text": candidate["normalized_text"],
        "memory_type": memory_type,
        "slot": candidate["slot"],
        "polarity": candidate["polarity"],
        "write_score": write_score,
        "extraction_reason": candidate["extraction_reason"],
        "update_hint": candidate["update_hint"],
        "metadata": {
            "source_utterance": content,
            "router_version": "write_router_v1",
            "source_turn_index": turn_index,
            "slot": candidate["slot"],
            "polarity": candidate["polarity"],
            "write_score": write_score,
            "extraction_reason": candidate["extraction_reason"],
            "update_hint": candidate["update_hint"],
        },
    }]


def extract_memories_from_episode(episode: dict[str, Any]) -> list[dict[str, Any]]:
    persona_id = str(episode["persona_id"])
    turns = episode.get("turns", [])

    extracted: list[dict[str, Any]] = []
    for turn in turns:
        extracted.extend(extract_memories_from_turn(turn, persona_id))
    return extracted
