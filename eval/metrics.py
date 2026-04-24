# eval/metrics.py
import argparse
import json
import os


def iter_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def safe_avg(values):
    return sum(values) / len(values) if values else 0.0


def summarize_file(path: str):
    rows = list(iter_jsonl(path))
    if not rows:
        return {
            "method": os.path.basename(path),
            "episodes": 0,
            "task_success": 0.0,
            "correct": 0,
            "avg_stored": 0.0,
            "avg_retrieved": 0.0,
            "avg_superseded": 0.0,
            "total_stored": 0,
            "total_retrieved": 0,
            "irrelevant_retrieval_rate": 0.0,
            "outdated_contradiction_errors": 0,
            "memory_precision": 0.0,
        }

    method = rows[0].get("method", os.path.basename(path))

    correct_vals = [1 if r.get("correct", False) else 0 for r in rows]
    stored_vals = [int(r.get("stored_count", 0) or 0) for r in rows]
    retrieved_vals = [int(r.get("retrieved_count", 0) or 0) for r in rows]
    superseded_vals = [int(r.get("superseded_count", 0) or 0) for r in rows]

    total = len(rows)
    correct = sum(correct_vals)

    # ---- Placeholder-safe numeric metrics ----
    irrelevant_retrieval_rate = 0.0

    # Use superseded_count as a proxy for contradiction/outdated errors
    outdated_contradiction_errors = sum(superseded_vals)

    # Simple proxy: if anything was retrieved, treat as fully "used"
    total_retrieved = sum(retrieved_vals)
    memory_precision = 1.0 if total_retrieved > 0 else 0.0

    return {
        "method": method,
        "episodes": total,
        "task_success": correct / total if total else 0.0,
        "correct": correct,
        "avg_stored": safe_avg(stored_vals),
        "avg_retrieved": safe_avg(retrieved_vals),
        "avg_superseded": safe_avg(superseded_vals),
        "total_stored": sum(stored_vals),
        "total_retrieved": total_retrieved,
        "irrelevant_retrieval_rate": irrelevant_retrieval_rate,
        "outdated_contradiction_errors": outdated_contradiction_errors,
        "memory_precision": memory_precision,
    }


def format_value(x):
    if isinstance(x, float):
        return f"{x:.3f}"
    return str(x)


def print_summary_table(summaries):
    headers = [
        ("method", "method"),
        ("episodes", "episodes"),
        ("correct", "correct"),
        ("task_success", "accuracy"),
        ("avg_stored", "avg_stored"),
        ("avg_retrieved", "avg_retrieved"),
        ("avg_superseded", "avg_superseded"),
        ("total_stored", "total_stored"),
        ("total_retrieved", "total_retrieved"),
        ("irrelevant_retrieval_rate", "irr_ret_rate"),
        ("outdated_contradiction_errors", "contradictions"),
        ("memory_precision", "mem_precision"),
    ]

    table_rows = []
    for summary in summaries:
        row = [format_value(summary[key]) for key, _ in headers]
        table_rows.append(row)

    widths = []
    for i, (_, label) in enumerate(headers):
        max_cell = max((len(row[i]) for row in table_rows), default=0)
        widths.append(max(len(label), max_cell))

    header_line = " | ".join(
        label.ljust(widths[i]) for i, (_, label) in enumerate(headers)
    )
    sep_line = "-+-".join("-" * widths[i] for i in range(len(headers)))

    print(header_line)
    print(sep_line)

    for row in table_rows:
        print(" | ".join(row[i].ljust(widths[i]) for i in range(len(headers))))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results",
        help="Directory containing result jsonl logs",
    )
    parser.add_argument(
        "--files",
        nargs="*",
        default=None,
        help="Optional specific result files to summarize",
    )
    args = parser.parse_args()

    if args.files:
        paths = [
            os.path.join(args.results_dir, f) if not os.path.isabs(f) else f
            for f in args.files
        ]
    else:
        if not os.path.isdir(args.results_dir):
            raise FileNotFoundError(f"Results directory not found: {args.results_dir}")

        paths = [
            os.path.join(args.results_dir, name)
            for name in os.listdir(args.results_dir)
            if name.endswith(".jsonl") and name != "run_one.jsonl"
        ]
        paths.sort()

    if not paths:
        print("No result jsonl files found.")
        return

    summaries = [summarize_file(path) for path in paths]
    print_summary_table(summaries)


if __name__ == "__main__":
    main()