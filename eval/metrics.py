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


def to_int(value, default=0):
    try:
        if value is None:
            return default
        return int(value)
    except (ValueError, TypeError):
        return default


def to_float(value, default=0.0):
    try:
        if value is None:
            return default
        return float(value)
    except (ValueError, TypeError):
        return default


def clean_label(value, default="UNKNOWN"):
    if value is None:
        return default
    s = str(value).strip()
    if s == "" or s.lower() in {"none", "nan"}:
        return default
    return s


def bucket_by_range(value, bins, labels, default="UNKNOWN"):
    if value is None:
        return default
    for i in range(len(bins) - 1):
        left = bins[i]
        right = bins[i + 1]
        if left <= value < right:
            return labels[i]
    return labels[-1]


def bucket_context_length_tokens(x):
    x = to_int(x, default=-1)
    if x < 0:
        return "UNKNOWN"
    bins = [0, 500, 2000, 5000, 10000, float("inf")]
    labels = ["0-499", "500-1999", "2000-4999", "5000-9999", "10000+"]
    return bucket_by_range(x, bins, labels)


def bucket_distance_tokens(x):
    x = to_int(x, default=-1)
    if x < 0:
        return "UNKNOWN"
    bins = [0, 100, 500, 1000, 5000, float("inf")]
    labels = ["0-99", "100-499", "500-999", "1000-4999", "5000+"]
    return bucket_by_range(x, bins, labels)


def bucket_distance_blocks(x):
    x = to_int(x, default=-1)
    if x < 0:
        return "UNKNOWN"
    bins = [0, 2, 5, 10, 20, float("inf")]
    labels = ["0-1", "2-4", "5-9", "10-19", "20+"]
    return bucket_by_range(x, bins, labels)


def bucket_irrelevant_tokens(x):
    x = to_int(x, default=-1)
    if x < 0:
        return "UNKNOWN"
    bins = [0, 100, 500, 1000, 5000, float("inf")]
    labels = ["0-99", "100-499", "500-999", "1000-4999", "5000+"]
    return bucket_by_range(x, bins, labels)


def bucket_proportion(x):
    x = to_float(x, default=-1.0)
    if x < 0:
        return "UNKNOWN"
    if x < 0.2:
        return "start_0.0_0.2"
    if x < 0.4:
        return "early_0.2_0.4"
    if x < 0.6:
        return "middle_0.4_0.6"
    if x < 0.8:
        return "late_0.6_0.8"
    return "end_0.8_1.0"


def bucket_end_index(x):
    x = to_int(x, default=-1)
    if x < 0:
        return "UNKNOWN"
    bins = [0, 20, 50, 100, 200, float("inf")]
    labels = ["0-19", "20-49", "50-99", "100-199", "200+"]
    return bucket_by_range(x, bins, labels)


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
            "irrelevant_retrieval_rate": "TODO",
            "outdated_contradiction_errors": "TODO",
            "memory_precision": "TODO",
        }

    method = rows[0].get("method", os.path.basename(path))
    correct_vals = [1 if r.get("correct", False) else 0 for r in rows]
    stored_vals = [to_int(r.get("stored_count", 0), 0) for r in rows]
    retrieved_vals = [to_int(r.get("retrieved_count", 0), 0) for r in rows]
    superseded_vals = [to_int(r.get("superseded_count", 0), 0) for r in rows]

    total = len(rows)
    correct = sum(correct_vals)

    summary = {
        "method": method,
        "episodes": total,
        "task_success": correct / total if total else 0.0,
        "correct": correct,
        "avg_stored": safe_avg(stored_vals),
        "avg_retrieved": safe_avg(retrieved_vals),
        "avg_superseded": safe_avg(superseded_vals),
        "total_stored": sum(stored_vals),
        "total_retrieved": sum(retrieved_vals),

        # Placeholders for later metrics
        "irrelevant_retrieval_rate": "TODO",
        "outdated_contradiction_errors": "TODO",
        "memory_precision": "TODO",
    }
    return summary


def build_group_summary(rows, group_key_fn):
    grouped = {}

    for r in rows:
        key = group_key_fn(r)
        key = clean_label(key, default="UNKNOWN")

        if key not in grouped:
            grouped[key] = {
                "episodes": 0,
                "correct": 0,
                "stored_total": 0,
                "retrieved_total": 0,
                "superseded_total": 0,
            }

        grouped[key]["episodes"] += 1
        grouped[key]["correct"] += 1 if r.get("correct", False) else 0
        grouped[key]["stored_total"] += to_int(r.get("stored_count", 0), 0)
        grouped[key]["retrieved_total"] += to_int(r.get("retrieved_count", 0), 0)
        grouped[key]["superseded_total"] += to_int(r.get("superseded_count", 0), 0)

    summary_rows = []
    for key, vals in grouped.items():
        episodes = vals["episodes"]
        summary_rows.append(
            {
                "group": key,
                "episodes": episodes,
                "correct": vals["correct"],
                "task_success": vals["correct"] / episodes if episodes else 0.0,
                "avg_stored": vals["stored_total"] / episodes if episodes else 0.0,
                "avg_retrieved": vals["retrieved_total"] / episodes if episodes else 0.0,
                "avg_superseded": vals["superseded_total"] / episodes if episodes else 0.0,
            }
        )

    summary_rows.sort(key=lambda x: (-x["episodes"], x["group"]))
    return summary_rows


def format_float(x):
    if isinstance(x, float):
        return f"{x:.3f}"
    return str(x)


def print_summary_table(summaries):
    headers = [
        "method",
        "episodes",
        "correct",
        "task_success",
        "avg_stored",
        "avg_retrieved",
        "avg_superseded",
        "total_stored",
        "total_retrieved",
        "irrelevant_retrieval_rate",
        "outdated_contradiction_errors",
        "memory_precision",
    ]

    table_rows = []
    for s in summaries:
        table_rows.append([format_float(s[h]) for h in headers])

    widths = []
    for i, h in enumerate(headers):
        max_cell = max([len(row[i]) for row in table_rows], default=0)
        widths.append(max(len(h), max_cell))

    header_line = " | ".join(h.ljust(widths[i]) for i, h in enumerate(headers))
    sep_line = "-+-".join("-" * widths[i] for i in range(len(headers)))

    print(header_line)
    print(sep_line)

    for row in table_rows:
        print(" | ".join(row[i].ljust(widths[i]) for i in range(len(headers))))


def print_group_table(title, rows):
    print(f"\n{title}")

    if not rows:
        print("No rows found.")
        return

    headers = [
        "group",
        "episodes",
        "correct",
        "task_success",
        "avg_stored",
        "avg_retrieved",
        "avg_superseded",
    ]

    table_rows = []
    for r in rows:
        table_rows.append([format_float(r[h]) for h in headers])

    widths = []
    for i, h in enumerate(headers):
        max_cell = max([len(row[i]) for row in table_rows], default=0)
        widths.append(max(len(h), max_cell))

    header_line = " | ".join(h.ljust(widths[i]) for i, h in enumerate(headers))
    sep_line = "-+-".join("-" * widths[i] for i in range(len(headers)))

    print(header_line)
    print(sep_line)

    for row in table_rows:
        print(" | ".join(row[i].ljust(widths[i]) for i in range(len(headers))))


def print_breakdowns_for_file(path: str):
    rows = list(iter_jsonl(path))
    if not rows:
        return

    method = rows[0].get("method", os.path.basename(path))
    print(f"\n{'=' * 80}")
    print(f"SCENARIO BREAKDOWNS: {method}")
    print(f"{'=' * 80}")

    breakdowns = [
        ("By topic", lambda r: r.get("topic")),
        ("By question_type", lambda r: r.get("question_type")),
        ("By persona_id", lambda r: r.get("persona_id")),
        (
            "By context_length_in_tokens bucket",
            lambda r: bucket_context_length_tokens(r.get("context_length_in_tokens")),
        ),
        (
            "By distance_to_ref_in_tokens bucket",
            lambda r: bucket_distance_tokens(r.get("distance_to_ref_in_tokens")),
        ),
        (
            "By distance_to_ref_in_blocks bucket",
            lambda r: bucket_distance_blocks(r.get("distance_to_ref_in_blocks")),
        ),
        (
            "By num_irrelevant_tokens bucket",
            lambda r: bucket_irrelevant_tokens(r.get("num_irrelevant_tokens")),
        ),
        (
            "By distance_to_ref_proportion_in_context bucket",
            lambda r: bucket_proportion(r.get("distance_to_ref_proportion_in_context")),
        ),
        (
            "By end_index_in_shared_context bucket",
            lambda r: bucket_end_index(r.get("end_index_in_shared_context")),
        ),
    ]

    for title, fn in breakdowns:
        group_rows = build_group_summary(rows, fn)
        print_group_table(title, group_rows)


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
    parser.add_argument(
        "--show_breakdowns",
        action="store_true",
        help="Print scenario breakdown tables for each result file",
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

    if args.show_breakdowns:
        for path in paths:
            print_breakdowns_for_file(path)


if __name__ == "__main__":
    main()
