# eval/run_eval.py
import json
import os
import re
import time
from datetime import datetime, timezone

from groq import Groq

from eval.scoring import is_correct, normalize_label


def iter_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


# Groq client
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))


def call_llm(prompt: str) -> str:
    """
    Call model via Groq and return text.
    Prints errors so we can debug instead of silently failing.
    """
    try:
        resp = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=8,
            temperature=0.0,
        )

        # Defensive parsing
        if not resp or not getattr(resp, "choices", None):
            print("[LLM ERROR] Empty response object")
            return ""

        msg = resp.choices[0].message if resp.choices else None
        content = (msg.content or "").strip() if msg else ""

        if not content:
            print("[LLM WARN] Model returned empty content")
        return content

    except Exception as e:
        print(f"[LLM ERROR] {type(e).__name__}: {e}")
        time.sleep(2)
        return ""


def extract_choice_label(text: str) -> str:
    """
    Convert model output into canonical label: (a)/(b)/(c)/(d).
    Handles outputs like:
      - "(c)"
      - "c"
      - "Answer: (c)"
      - "The answer is c"
    Returns "" if not found.
    """
    if not text:
        return ""

    s = text.strip().lower()

    # Prefer explicit (a)/(b)/(c)/(d)
    m = re.search(r"\(([a-d])\)", s)
    if m:
        return f"({m.group(1)})"

    # Fallback to standalone letter
    m = re.search(r"\b([a-d])\b", s)
    if m:
        return f"({m.group(1)})"

    # Final fallback: remove non a-d and take first char if any remains
    letter = re.sub(r"[^a-d]", "", s)[:1]
    return f"({letter})" if letter else ""


class NoMemoryGroq:
    def __init__(
        self,
        sleep_s: float = 1.5,
        context_cap_chars: int = 8000,
        options_cap_chars: int = 2500,
    ):
        self.sleep_s = sleep_s
        self.context_cap_chars = context_cap_chars
        self.options_cap_chars = options_cap_chars

    def build_prompt(self, episode: dict) -> str:
        context = (episode.get("sliced_context", "") or "")[-self.context_cap_chars:]
        options = (episode.get("options", "") or "")[:self.options_cap_chars]
        question = episode.get("question", "") or ""

        # Stricter format request (helps compliance a lot)
        return (
            f"Conversation history:\n{context}\n\n"
            f"Question: {question}\n"
            f"Options: {options}\n"
            f"Output format must be EXACTLY one of: (a) (b) (c) (d). No other text."
        )

    def answer(self, episode: dict) -> tuple[str, str]:
        """
        Returns:
          (prediction_label, raw_model_text)
        """
        prompt = self.build_prompt(episode)
        raw = call_llm(prompt)

        pred = extract_choice_label(raw)

        time.sleep(self.sleep_s)
        return pred, raw


def main():
    os.makedirs("results", exist_ok=True)

    episodes_path = os.path.join("data", "dev_10.jsonl")
    out_path = os.path.join("results", "no_memory_dev10.jsonl")

    method = NoMemoryGroq()

    correct = 0
    total = 0

    with open(out_path, "w", encoding="utf-8") as out:
        for ep in iter_jsonl(episodes_path):
            pred_label, raw_text = method.answer(ep)
            gold_raw = ep.get("answer", "")

            # Debug first 2 episodes so you can see what the model returned
            if ep.get("episode_id") in [0, 1]:
                print(f"[DEBUG ep {ep.get('episode_id')}] raw_model_text={repr(raw_text)} pred={pred_label} gold={gold_raw}")

            rec = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "episode_id": ep.get("episode_id"),
                "method": "no_memory_groq_openai_gpt-oss-120b",
                "prediction_raw": raw_text,          # store the real raw model output
                "prediction": normalize_label(pred_label),
                "gold_raw": gold_raw,
                "gold": normalize_label(gold_raw),
                "correct": is_correct(pred_label, gold_raw),

                # placeholders for later
                "retrieved": [],
                "retrieved_scores": [],
                "stored_count": 0,
                "retrieved_count": 0,
                "superseded_count": 0,

                # metadata for later breakdowns
                "question_type": ep.get("question_type"),
                "topic": ep.get("topic"),
            }

            out.write(json.dumps(rec) + "\n")
            total += 1
            correct += int(rec["correct"])

    acc = correct / total if total else 0.0
    print(f"Accuracy: {correct}/{total} = {acc:.3f}")
    print(f"Wrote logs to: {out_path}")


if __name__ == "__main__":
    main()