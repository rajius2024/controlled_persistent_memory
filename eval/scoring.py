import re

def normalize_label(x: str) -> str:
    """Return '(a)'..'(d)' if found, else ''."""
    if x is None:
        return ""
    s = str(x).strip().lower()
    m = re.search(r"\(([a-d])\)", s)
    if m:
        return f"({m.group(1)})"
    # handle bare a/b/c/d
    m = re.search(r"\b([a-d])\b", s)
    if m:
        return f"({m.group(1)})"
    return ""

def is_correct(pred: str, gold: str) -> bool:
    return normalize_label(pred) == normalize_label(gold)