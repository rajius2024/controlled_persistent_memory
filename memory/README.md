```markdown
# Memory Subsystem

Controlled persistent memory layer for local PersonaMem-style evaluation. Handles memory extraction, storage, conflict resolution, and filtered retrieval to reduce stale or contradictory context reaching the answer prompt.

---

## What it does

Unlike vanilla retrieval (which stores all context and retrieves by similarity), this subsystem is selective:

- only user preference statements matching extraction triggers are written
- conflicting older memories are marked `superseded` rather than deleted
- retrieval considers only `active` memories, filtered by `persona_id`
- retrieved memories are injected directly into the answer prompt

---

## Directory overview

| File | Purpose |
|---|---|
| `schema.py` | `MemoryRecord` dataclass and lifecycle state |
| `store_interface.py` | Backend-agnostic storage contract |
| `chroma_store.py` | Chroma implementation of the storage contract |
| `write_router_v0.py` | Rule-based extraction and light normalization of user turns |
| `versioning.py` | Conflict detection and supersession logic |
| `controlled_store_builder.py` | Orchestrates per-episode write path end to end |
| `logging_utils.py` | JSONL logs for writes, supersessions, retrievals, and summaries |
| `demo_versioning.py` | Local demo for validating supersession without the full eval runner |

---

## Data model

Each stored memory is a `MemoryRecord` with the following fields:

| Field | Description |
|---|---|
| `memory_id` | Unique record identifier |
| `persona_id` | Owner of the memory |
| `turn_index` | Source turn position in the conversation |
| `status` | `active` or `superseded` |
| `normalized_mem_text` | Canonicalized memory text used for retrieval |
| `memory_type` | Semantic category used for filtering and conflict grouping |
| `time_marker` | UTC timestamp set at record creation |
| `supersession_link` | ID of the older memory this record replaced (when applicable) |
| `metadata` | Additional source-level annotations |

**Status semantics**

- `active` — eligible for retrieval
- `superseded` — retained for audit and debugging, excluded from retrieval

---

## Storage

The current backend is Chroma. Each record is stored as:

- **id** → `memory_id`
- **document** → `normalized_mem_text`
- **metadata** → `persona_id`, `turn_index`, `status`, `memory_type`, `time_marker`, `supersession_link`

Only flat top-level fields are stored in metadata to keep filtering predictable. Persistent state is created per run directory:

```text
results/controlled_runs/episode_0/chromadb/
results/controlled_runs/episode_0/logs/
```

The backend-agnostic interface exposes: `put_memory`, `get_memory_by_id`, `list_memories`, `query_memories`, `supersede_memory`.

---

## Write path

```
Episode turns
    │
    ▼
write_router_v0       ← inspect user turns only, apply extraction triggers
    │
    ▼
candidate MemoryRecord(s)
    │
    ▼
versioning.add_memory(...)
    │
    ├── mark older conflicting memory as superseded
    │
    ▼
ChromaMemoryStore.put_memory(...)
    │
    ▼
active memory set
```

**Extraction triggers** — the router matches preference-style language: `prefer`, `like`, `love`, `don't like`, `favorite`, `from now on`, `enjoy`, `interested in`.

**Normalization examples**

| Raw | Normalized |
|---|---|
| I like tea. | likes tea |
| I prefer coffee now. | prefers coffee now |
| I don't like phone calls. | dislikes phone calls |
| I am interested in ambient music. | interested in ambient music |

The router is intentionally conservative — it favors cleaner writes over high recall.

**Conflict and supersession** — two memories conflict when they share the same `persona_id`, the older one is `active`, they fall into the same conflict bucket (by `memory_type`, or a text-derived fallback), and they differ after normalization. When conflict is detected, the older memory is marked `superseded` and the new memory stores its ID in `supersession_link`.

---

## Retrieval path

```
active memories for persona
    │
    ▼
optional filter by memory_type
    │
    ▼
embed active memory texts + embed question
    │
    ▼
rank by similarity → return top-k active memories
    │
    ▼
build answer prompt
```

The prompt contains only retrieved active memories, the question, options, and a strict answer-format instruction — not raw conversation turns.

---

## Logging

Per-episode JSONL logs are written to `results/controlled_runs/episode_N/logs/`.

| File | Contents |
|---|---|
| `writes.jsonl` | One line per stored memory (episode id, persona id, turn index, memory id, text, type, conflict bucket) |
| `supersedes.jsonl` | One line per supersession event (old id, new id, persona id, turn index) |
| `retrievals.jsonl` | One line per retrieval call (query, active ids, retrieved ids, texts, scores, count) |
| `episode_summary.jsonl` | Per-episode totals: stored, active, superseded counts, retrieved memories, prediction, gold label |

---

## Current status and known limitations

On the current 10-episode local slice, episodes are clustered around the same persona and topic region. The router extracts one stable music-preference memory, so retrieval returns the same active memory consistently. This reflects conservative write coverage, not a storage or retrieval failure.

| Limitation | Notes |
|---|---|
| Narrow router coverage | Softer or implicit preference statements are still missed |
| Heuristic conflict grouping | Simple type-bucket approach, should be refined |
| Versioning not stressed | The 10-episode slice generates few contradictory updates |
| Retrieval diversity | Bounded by stored memory diversity |

---

## Quick commands

```bash
# Validate supersession behavior
python -m memory.demo_versioning

# Build episodes
python -m data.build_episode

# Run controlled evaluation
PYTHONNOUSERSITE=1 python -m eval.run_eval --method controlled --top_k 5 --episodes_path data/dev_latest.jsonl
```
```
