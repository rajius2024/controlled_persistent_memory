"""
Shared LLM Utility
==================
Single call_llm() used by ALL methods and eval scripts.

Everyone imports from here:
    from utils.llm import call_llm, normalize_label, is_correct

This ensures:
- Same model across all methods (fair comparison)
- Same max_tokens, temperature settings
- Same answer parsing logic
- No duplicate code across runner.py, vanilla_rag.py, eval/run_eval.py
"""

import os
import re
import time

from groq import Groq

# ── Config ────────────────────────────────────────────────────────────────────
MODEL            = "meta-llama/llama-4-scout-17b-16e-instruct"
MAX_TOKENS       = 8
TEMPERATURE      = 0.0
MAX_RETRIES      = 3
RATE_LIMIT_SLEEP = 5.0

_client = None

def _get_client():
    global _client
    if _client is None:
        _client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    return _client


# ── LLM Call ──────────────────────────────────────────────────────────────────

def call_llm(prompt: str, system: str = None) -> str:
    """
    Call Llama 3.3-70b via Groq with retry on rate limit errors.

    Args:
        prompt: User prompt string
        system: Optional system message (default forces single-letter answer)

    Returns:
        Raw model response string, or "" on failure
    """
    if system is None:
        system = (
            "You are a helpful assistant answering multiple-choice questions. "
            "Output format must be EXACTLY one of: (a) (b) (c) (d). No other text."
        )

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    for attempt in range(MAX_RETRIES):
        try:
            resp = _get_client().chat.completions.create(
                model=MODEL,
                messages=messages,
                max_tokens=MAX_TOKENS,
                temperature=TEMPERATURE,
            )
            if not resp or not getattr(resp, "choices", None):
                print("[LLM ERROR] Empty response object")
                return ""
            content = (resp.choices[0].message.content or "").strip()
            if not content:
                print("[LLM WARN] Model returned empty content")
            return content

        except Exception as e:
            err = str(e)
            print(f"  [DEBUG] Exception type: {type(e).__name__}, message: {err[:200]}")
            if "413" in err or "429" in err or "rate_limit" in err.lower():
                wait = 15 * (attempt + 1)
                print(f"  [RATE LIMIT] Waiting {wait}s... (attempt {attempt+1}/{MAX_RETRIES})")
                time.sleep(wait)
            elif "401" in err:
                print("  [AUTH ERROR] Set GROQ_API_KEY with: export GROQ_API_KEY=...")
                return ""
            else:
                print(f"  [LLM ERROR] {type(e).__name__}: {e}")
                return ""

    print(f"  [FAILED] All {MAX_RETRIES} retries exhausted.")
    return ""


# ── Answer Parsing ─────────────────────────────────────────────────────────────

def normalize_label(text: str) -> str:
    """
    Convert any answer format to canonical (a)/(b)/(c)/(d).

    Handles:
        "(c)" -> "(c)"
        "c"   -> "(c)"
        "Answer: (c)" -> "(c)"
        "(C)" -> "(c)"
    Returns "" if no valid letter found.
    """
    if not text:
        return ""
    s = str(text).strip().lower()

    # Prefer explicit (a)/(b)/(c)/(d)
    m = re.search(r"\(([a-d])\)", s)
    if m:
        return f"({m.group(1)})"

    # Fallback to standalone letter
    m = re.search(r"\b([a-d])\b", s)
    if m:
        return f"({m.group(1)})"

    # Final fallback
    letter = re.sub(r"[^a-d]", "", s)[:1]
    return f"({letter})" if letter else ""


def is_correct(pred: str, gold: str) -> bool:
    """Compare predicted and gold answers after normalization."""
    return normalize_label(pred) == normalize_label(gold)


# ── Rate limit helper ─────────────────────────────────────────────────────────

def sleep_between_calls(seconds: float = RATE_LIMIT_SLEEP):
    """Sleep between LLM calls to respect rate limits."""
    time.sleep(seconds)


# ── Quick test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Testing call_llm()...")
    resp = call_llm("Reply with just: (a)")
    print(f"Raw response  : {resp!r}")
    print(f"Normalized    : {normalize_label(resp)}")
    print(f"Is correct (a): {is_correct(resp, '(a)')}")

