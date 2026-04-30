from pathlib import Path
import os

p = Path("eval/methods/controlled.py")
s = p.read_text()

# Add os import
if "import os" not in s.split("\n")[:25]:
    if "import math\n" in s:
        s = s.replace("import math\n", "import math\nimport os\n", 1)
    else:
        s = s.replace("from __future__ import annotations\n", "from __future__ import annotations\n\nimport os\n", 1)

# Add ablation config
target = "        self.last_trace: dict[str, Any] | None = None\n"
replacement = '''        self.last_trace: dict[str, Any] | None = None

        self.ablation = os.environ.get("CONTROLLED_ABLATION", "full").strip().lower()
        valid_ablations = {"full", "no_bm25", "no_options", "no_superseded"}
        if self.ablation not in valid_ablations:
            raise ValueError(
                f"Unsupported CONTROLLED_ABLATION={self.ablation!r}. "
                f"Expected one of {sorted(valid_ablations)}."
            )
        print(f"[controlled] ablation={self.ablation}", flush=True)
'''
if "self.ablation = os.environ.get" not in s:
    s = s.replace(target, replacement, 1)

# Disable superseded retrieval
old_sup = '''        if question_type in EVOLUTION_QUESTION_TYPES:
            superseded_memories = store.list_memories(
                persona_id=persona_id,
                status=MemoryStatus.SUPERSEDED.value,
                memory_type=self.memory_type,
            )
        else:
            superseded_memories = []
'''
new_sup = '''        if (
            question_type in EVOLUTION_QUESTION_TYPES
            and self.ablation != "no_superseded"
        ):
            superseded_memories = store.list_memories(
                persona_id=persona_id,
                status=MemoryStatus.SUPERSEDED.value,
                memory_type=self.memory_type,
            )
        else:
            superseded_memories = []
'''
if 'self.ablation != "no_superseded"' not in s:
    s = s.replace(old_sup, new_sup, 1)

# Disable BM25 retrieval
old_bm25 = '''        bm25_active_records = self._bm25_retrieve(filtered_active, query=query, k=20)
        bm25_active_rows = self._scored_candidates(
            bm25_active_records,
            query=query,
            cache_key_prefix=f"controlled_bm25_active_{episode.get('question_id', episode.get('episode_id'))}",
            episode=episode,
        )
        bm25_active_rows = sorted(bm25_active_rows, key=lambda r: (r["score"], r["memory"].memory_id), reverse=True)
'''
new_bm25 = '''        if self.ablation == "no_bm25":
            bm25_active_rows = []
        else:
            bm25_active_records = self._bm25_retrieve(filtered_active, query=query, k=20)
            bm25_active_rows = self._scored_candidates(
                bm25_active_records,
                query=query,
                cache_key_prefix=f"controlled_bm25_active_{episode.get('question_id', episode.get('episode_id'))}",
                episode=episode,
            )
            bm25_active_rows = sorted(bm25_active_rows, key=lambda r: (r["score"], r["memory"].memory_id), reverse=True)
'''
if 'self.ablation == "no_bm25"' not in s:
    s = s.replace(old_bm25, new_bm25, 1)

# Disable option-conditioned retrieval
old_opt = '''        option_rows = self._option_conditioned_rows(
            filtered_active,
            question=question,
            options=options,
            episode=episode,
            per_option_k=4,
        )
        option_rows = sorted(option_rows, key=lambda r: (r["score"], r["memory"].memory_id), reverse=True)
'''
new_opt = '''        if self.ablation == "no_options":
            option_rows = []
        else:
            option_rows = self._option_conditioned_rows(
                filtered_active,
                question=question,
                options=options,
                episode=episode,
                per_option_k=4,
            )
            option_rows = sorted(option_rows, key=lambda r: (r["score"], r["memory"].memory_id), reverse=True)
'''
if 'self.ablation == "no_options"' not in s:
    s = s.replace(old_opt, new_opt, 1)

# Add ablation to trace
old_trace = '''            "topic": episode.get("topic", ""),
        }
'''
new_trace = '''            "topic": episode.get("topic", ""),
            "ablation": self.ablation,
        }
'''
if '"ablation": self.ablation' not in s:
    s = s.replace(old_trace, new_trace, 1)

p.write_text(s)
print("[OK] Added CONTROLLED_ABLATION switches.")
