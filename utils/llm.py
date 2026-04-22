"""
Shared LLM Utility
==================
Single call_llm() used by ALL methods and eval scripts.
"""

import os
import re
import time

from groq import Groq

MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"
MAX_TOKENS = 12
TEMPERATURE = 0.0
MAX_RETRIES = 4
RATE_LIMIT_SLEEP = 5.0

_client = None


def _get_client():
    global _client
    if _client is None:
        _client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    return _client


def normalize_label(text: str) -> str:
    if not text:
        return ""
    s = str(text).strip().lower()

    m = re.search(r"\(([a-d])\)", s)
    if m:
        return f"({m.group(1)})"

    m = re.search(r"\b([a-d])\b", s)
    if m:
        return f"({m.group(1)})"

    letter = re.sub(r"[^a-d]", "", s)[:1]
    return f"({letter})" if letter else ""


def is_correct(pred: str, gold: str) -> bool:
    return normalize_label(pred) == normalize_label(gold)


def call_llm(prompt: str, system: str = None) -> str:
    if system is None:
        system = (
            "You are answering a multiple-choice question. "
            "Return EXACTLY one of: (a) (b) (c) (d). "
            "Do not return any other text."
        )

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": prompt},
    ]

    last_content = ""

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
                time.sleep(1.0 * (attempt + 1))
                continue

            content = (resp.choices[0].message.content or "").strip()
            last_content = content

            if not content:
                print("[LLM WARN] Empty content")
                time.sleep(1.0 * (attempt + 1))
                continue

            if normalize_label(content):
                return content

            print(f"[LLM WARN] Malformed answer output: {content!r}")
            time.sleep(1.0 * (attempt + 1))

        except Exception as e:
            err = str(e)
            print(f"[DEBUG] Exception type: {type(e).__name__}, message: {err[:200]}")

            if "413" in err or "429" in err or "rate_limit" in err.lower():
                wait = 15 * (attempt + 1)
                print(f"[RATE LIMIT] Waiting {wait}s... (attempt {attempt + 1}/{MAX_RETRIES})")
                time.sleep(wait)
            elif "401" in err:
                print("[AUTH ERROR] Set GROQ_API_KEY with: export GROQ_API_KEY=...")
                return ""
            else:
                print(f"[LLM ERROR] {type(e).__name__}: {e}")
                return ""

    print(f"[FAILED] All {MAX_RETRIES} retries exhausted. Last content={last_content!r}")
    return last_content


def sleep_between_calls(seconds: float = RATE_LIMIT_SLEEP):
    time.sleep(seconds)


if __name__ == "__main__":
    print("Testing call_llm()...")
    resp = call_llm("Reply with just: (a)")
    print(f"Raw response  : {resp!r}")
    print(f"Normalized    : {normalize_label(resp)}")
    print(f"Is correct (a): {is_correct(resp, '(a)')}")
