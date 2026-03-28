import json, os
from datetime import datetime, timezone
datetime.now(timezone.utc).isoformat()
from eval.scoring import is_correct, normalize_label
from eval.methods.no_memory_stub import NoMemoryStub

def load_prompt(path="prompts/base_prompt.txt") -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def main():
    os.makedirs("results", exist_ok=True)

    with open("data/sample_view.json", "r", encoding="utf-8") as f:
        ep = json.load(f)

    prompt_template = load_prompt()
    method = NoMemoryStub()

    pred_raw = method.answer(ep, prompt_template)
    gold_raw = ep.get("answer", "")

    record = {
        "timestamp": datetime.utcnow().isoformat(),
        "episode_id": ep.get("episode_id"),
        "method": "no_memory_stub",
        "prediction_raw": pred_raw,
        "prediction": normalize_label(pred_raw),
        "gold_raw": gold_raw,
        "gold": normalize_label(gold_raw),
        "correct": is_correct(pred_raw, gold_raw),

        # placeholders for Day 3+
        "retrieved": [],
        "retrieved_scores": [],
        "stored_count": 0,
        "retrieved_count": 0,
        "superseded_count": 0,

        # handy metadata (already present in episode)
        "question_type": ep.get("question_type"),
        "topic": ep.get("topic"),
    }

    with open("results/run_one.jsonl", "a", encoding="utf-8") as out:
        out.write(json.dumps(record) + "\n")

    print(record)

if __name__ == "__main__":
    main()