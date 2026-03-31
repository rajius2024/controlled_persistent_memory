from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from memory.schema import MemoryRecord, MemoryStatus
from memory.store_interface import MemoryStore


@dataclass
class VersioningResult:
    stored_record: MemoryRecord
    superseded_ids: list[str]
    conflict_bucket: Optional[str] = None


_NEGATION_TOKENS = {"don't", "dont", "do not", "dislike", "hate", "avoid", "no"}


def _status_value(value: object) -> str:
    if isinstance(value, MemoryStatus):
        return value.value
    return str(value)


def _is_negative(text: str) -> bool:
    lowered = text.lower()
    return any(token in lowered for token in _NEGATION_TOKENS)


def _bucket_from_text(text: str) -> str:
    """
    Very simple v0 bucket for conflict grouping.
    We only need something stable enough for local controlled-memory behavior.
    """
    lowered = text.lower()

    if "favorite color" in lowered or "color" in lowered:
        return "color_preference"
    if "coffee" in lowered or "tea" in lowered or "drink" in lowered:
        return "drink_preference"
    if "food" in lowered or "pizza" in lowered or "pasta" in lowered or "spicy" in lowered:
        return "food_preference"
    if "phone" in lowered or "call" in lowered or "text" in lowered or "email" in lowered:
        return "communication_preference"
    if "morning" in lowered or "night" in lowered or "weekend" in lowered:
        return "schedule_preference"

    return "general_preference"


def conflict_bucket(record: MemoryRecord) -> str:
    if record.memory_type and record.memory_type != "general_preference":
        return record.memory_type
    return _bucket_from_text(record.normalized_mem_text)


def memories_conflict(existing: MemoryRecord, new: MemoryRecord) -> bool:
    """
    v0 conflict rule:
    - same persona
    - both active
    - same conflict bucket
    - not identical text
    """
    if existing.persona_id != new.persona_id:
        return False

    if _status_value(existing.status) != MemoryStatus.ACTIVE.value:
        return False

    existing_bucket = conflict_bucket(existing)
    new_bucket = conflict_bucket(new)

    if existing_bucket != new_bucket:
        return False

    if existing.normalized_mem_text.strip().lower() == new.normalized_mem_text.strip().lower():
        return False

    # For v0, treat same bucket with changed content as a superseding update.
    return True


def add_memory(record: MemoryRecord, store: MemoryStore) -> VersioningResult:
    """
    Add a new memory and supersede old conflicting active memories.

    Behavior:
    - find active records for same persona and type
    - supersede conflicts
    - new record links to the most recent conflicting memory, if any
    - then store the new record
    """
    active_candidates = store.list_memories(
        persona_id=record.persona_id,
        status=MemoryStatus.ACTIVE.value,
        memory_type=record.memory_type,
    )

    conflicts = [m for m in active_candidates if memories_conflict(m, record)]

    superseded_ids: list[str] = []
    replacement_link: Optional[str] = record.supersession_link

    if conflicts:
        # Use most recent conflicting record as the forward link target.
        most_recent_conflict = max(conflicts, key=lambda m: m.turn_index)
        replacement_link = most_recent_conflict.memory_id

        for old in conflicts:
            store.supersede_memory(old.memory_id, replacement_id=record.memory_id)
            superseded_ids.append(old.memory_id)

    final_record = record.model_copy(
        update={
            "status": MemoryStatus.ACTIVE,
            "supersession_link": replacement_link,
        }
    )
    stored = store.put_memory(final_record)

    return VersioningResult(
        stored_record=stored,
        superseded_ids=superseded_ids,
        conflict_bucket=conflict_bucket(final_record),
    )
