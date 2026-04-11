from memory.schema import MemoryRecord, MemoryStatus
from memory.versioning import add_memory
from memory.chroma_store import ChromaMemoryStore

store = ChromaMemoryStore(run_dir="tmp/versioning_test", collection_name="test")

m1 = MemoryRecord(
    memory_id="m1",
    persona_id="p1",
    turn_index=0,
    status=MemoryStatus.ACTIVE,
    normalized_mem_text="likes tea",
    memory_type="drink_preference",
    metadata={"slot": "drink", "polarity": "positive", "write_score": 3, "extraction_reason": "like_pattern", "update_hint": False},
)

m2 = MemoryRecord(
    memory_id="m2",
    persona_id="p1",
    turn_index=1,
    status=MemoryStatus.ACTIVE,
    normalized_mem_text="dislikes tea",
    memory_type="drink_preference",
    metadata={"slot": "drink", "polarity": "negative", "write_score": 3, "extraction_reason": "negative_preference_pattern", "update_hint": False},
)

print(add_memory(m1, store))
print(add_memory(m2, store))
print(store.list_memories(persona_id="p1"))
