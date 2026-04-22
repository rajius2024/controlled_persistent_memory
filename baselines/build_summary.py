import os
import json
import time
from groq import Groq

INPUT_FILE = os.path.join("data", "dev_latest.jsonl")
OUTPUT_FILE = os.path.join("data", "dev_latest_with_summary.jsonl")

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

MODEL = "llama-3.3-70b-versatile"

def generate_summary(context):
    prompt = (
        "Summarize this conversation history into a concise persona profile. "
        "Focus on facts about the user and their preferences.\n\n"
        f"History:\n{context[-4000:]}"
    )
    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            err = str(e)
            if "429" in err or "rate_limit" in err.lower():
                wait = 15 * (attempt + 1)
                print(f"  [RATE LIMIT] Waiting {wait}s...")
                time.sleep(wait)
            else:
                print(f"  [ERROR] {e}")
                return ""
    return ""

def main():
    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found.")
        return

    with open(INPUT_FILE, 'r') as f:
        lines = f.readlines()

    results = []
    print(f"Generating summaries for {len(lines)} episodes using {MODEL}...")
    for i, line in enumerate(lines):
        ep = json.loads(line)
        ep['summary_memory'] = generate_summary(ep.get('sliced_context', ''))
        results.append(ep)
        if (i + 1) % 10 == 0:
            print(f"Progress: {i+1}/{len(lines)}")
        time.sleep(1.0)

    with open(OUTPUT_FILE, 'w') as f:
        for item in results:
            f.write(json.dumps(item) + "\n")
    print(f"Success! Saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
