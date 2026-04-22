from __future__ import annotations

from memory.chroma_store import ChromaMemoryStore
from memory.schema import MemoryRecord
from memory.versioning import add_memory


def main() -> None:
    store = ChromaMemoryStore(run_dir="runs/demo_versioning", collection_name="demo_versioning")

    first = MemoryRecord(
        memory_id="m1",
        persona_id="p1",
        turn_index=0,
        normalized_mem_text="prefers tea",
        memory_type="drink_preference",
    )
    second = MemoryRecord(
        memory_id="m2",
        persona_id="p1",
        turn_index=2,
        normalized_mem_text="prefers coffee",
        memory_type="drink_preference",
    )

    result_1 = add_memory(first, store)
    print("Stored first:", result_1.stored_record)

    result_2 = add_memory(second, store)
    print("Stored second:", result_2.stored_record)
    print("Superseded ids:", result_2.superseded_ids)

    active = store.list_memories(persona_id="p1", status="active")
    superseded = store.list_memories(persona_id="p1", status="superseded")

    print("\nActive memories:")
    for item in active:
        print("-", item.memory_id, item.normalized_mem_text, item.status)

    print("\nSuperseded memories:")
    for item in superseded:
        print("-", item.memory_id, item.normalized_mem_text, item.status)


if __name__ == "__main__":
    main()
