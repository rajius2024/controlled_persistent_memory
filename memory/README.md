# Memory subsystem v0

Implemented:
- `schema.py`
- `store_interface.py`

## MemoryRecord
Fields:
- `memory_id`
- `persona_id`
- `turn_index`
- `status`
- `normalized_mem_text`
- `memory_type`
- `time_marker`
- `supersession_link`
- `metadata`

## MemoryStore
Current backend-agnostic interface methods:
- `put_memory`
- `get_memory_by_id`
- `list_memories`
- `query_memories`
- `supersede_memory`

## Notes
- `turn_index` is the conversation-order field.
- `time_marker` is a runtime/audit UTC timestamp.
- `supersession_link` points from a new memory to the old memory it supersedes.

## Not implemented yet
- Chroma backend
- write router
- versioning behavior
- controlled retrieval behavior
