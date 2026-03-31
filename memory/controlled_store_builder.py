from __future__ import annotations

from typing import Any

from memory.logging_utils import log_supersede_event, log_write_event
from memory.schema import MemoryRecord
from memory.store_interface import MemoryStore
from memory.versioning import add_memory
from memory.write_router_v0 import extract_memories_from_turn


def build_controlled_store_from_episode(
    episode: dict[str, Any],
    store: MemoryStore,
    run_dir: str | None = None,
) -> dict[str, Any]:
    """
    Process turns sequentially:
    - write router proposes memories
    - versioning decides supersession
    - store persists active records
    """
    persona_id = str(episode["persona_id"])
    turns = episode.get("turns", [])

    stored_records: list[MemoryRecord] = []
    superseded_ids: list[str] = []

    for turn in turns:
        candidate_records = extract_memories_from_turn(turn, persona_id)

        for record in candidate_records:
            result = add_memory(record, store)
            stored_records.append(result.stored_record)
            superseded_ids.extend(result.superseded_ids)

            if run_dir is not None:
                log_write_event(
                    run_dir,
                    {
                        "episode_id": episode["episode_id"],
                        "persona_id": persona_id,
                        "turn_index": record.turn_index,
                        "stored_memory_id": result.stored_record.memory_id,
                        "stored_text": result.stored_record.normalized_mem_text,
                        "memory_type": result.stored_record.memory_type,
                        "conflict_bucket": result.conflict_bucket,
                    },
                )

                for old_id in result.superseded_ids:
                    log_supersede_event(
                        run_dir,
                        {
                            "episode_id": episode["episode_id"],
                            "persona_id": persona_id,
                            "turn_index": record.turn_index,
                            "old_memory_id": old_id,
                            "new_memory_id": result.stored_record.memory_id,
                            "new_text": result.stored_record.normalized_mem_text,
                        },
                    )

    return {
        "stored_records": stored_records,
        "stored_count": len(stored_records),
        "superseded_ids": superseded_ids,
        "superseded_count": len(superseded_ids),
    }
