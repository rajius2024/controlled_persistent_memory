from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def ensure_run_dirs(run_dir: str) -> Path:
    run_path = Path(run_dir)
    logs_path = run_path / "logs"
    logs_path.mkdir(parents=True, exist_ok=True)
    return logs_path


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def log_event(run_dir: str, log_name: str, payload: dict[str, Any]) -> None:
    logs_path = ensure_run_dirs(run_dir)
    enriched = {
        "logged_at": datetime.now(timezone.utc).isoformat(),
        **payload,
    }
    _append_jsonl(logs_path / f"{log_name}.jsonl", enriched)


def log_write_event(run_dir: str, payload: dict[str, Any]) -> None:
    log_event(run_dir, "writes", payload)


def log_supersede_event(run_dir: str, payload: dict[str, Any]) -> None:
    log_event(run_dir, "supersedes", payload)


def log_retrieval_event(run_dir: str, payload: dict[str, Any]) -> None:
    log_event(run_dir, "retrievals", payload)


def log_episode_summary(run_dir: str, payload: dict[str, Any]) -> None:
    log_event(run_dir, "episode_summary", payload)
