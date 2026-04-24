"""
Shared LLM Utility
==================
Single call_llm() used by ALL methods and eval scripts.
Runs against a local pinned open-source model served on Sol.
"""

import os
import re
import time

import requests

MODEL = os.environ.get(
    "LOCAL_LLM_MODEL",
    "/scratch/vnaruvan/models/llama31_8b_instruct_pinned",
)
BASE_URL = os.environ.get(
    "LOCAL_LLM_BASE_URL",
    "http://127.0.0.1:8000/v1/chat/completions",
)
API_KEY = os.environ.get("LOCAL_LLM_API_KEY", "local-token")

MAX_TOKENS = 16
TEMPERATURE = 0.0
MAX_RETRIES = 4
RATE_LIMIT_SLEEP = 5.0


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

    last_content = ""

    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.post(
                BASE_URL,
                headers={
                    "Authorization": f"Bearer {API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": MODEL,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": prompt},
                    ],
                    "temperature": TEMPERATURE,
                    "max_tokens": MAX_TOKENS,
                },
                timeout=180,
            )
            resp.raise_for_status()
            data = resp.json()

            content = (data["choices"][0]["message"]["content"] or "").strip()
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

            if "429" in err or "rate_limit" in err.lower():
                wait = 15 * (attempt + 1)
                print(f"[RATE LIMIT] Waiting {wait}s... (attempt {attempt + 1}/{MAX_RETRIES})")
                time.sleep(wait)
            else:
                print(f"[LLM ERROR] {type(e).__name__}: {e}")
                time.sleep(1.0 * (attempt + 1))

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