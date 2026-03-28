"""
No-Memory Baseline Runner
=========================
Lead: Aravindan Chidambaram

What this does:
- Loads 10 pre-built episodes from data/dev_10.jsonl
- For each episode, sends the raw sliced conversation context + question to Groq (Llama 3.3)
- Records whether the predicted answer matches the correct answer
- Reports overall accuracy

Usage:
    export GROQ_API_KEY="your_key_here"
    python runner.py

Results (dev_10, no memory):
    Accuracy: 6/10 = 60.00%

Dependencies:
    pip install groq
    
Notes:
- Context is truncated to last 8000 chars to stay under Groq's 12k token limit
- Answer matching strips parentheses before comparing (e.g. '(c)' == 'c')
- Rate limit: 1.5s sleep between requests
"""
import os
import json
import time
import re 
from groq import Groq
from data.load_data import load_questions, load_shared_contexts
from data.build_episode import build_episodes

# Load API key from environment
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

def call_llm(prompt: str) -> str:
    """Call Llama 3.3 via Groq."""
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=10,
            temperature=0.0,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"[LLM ERROR] {e}")
        time.sleep(5)
        return ""


def run_no_memory(episodes):
    correct = 0
    for ep in episodes:
        # Truncate context to ~8000 chars to stay under token limit
        context = ep['sliced_context'][-8000:]
        
        prompt = (
            f"Conversation history:\n{context}\n\n"
            f"Question: {ep['question']}\n"
            f"Options: {ep['options']}\n"
            f"Reply with just the answer letter (a/b/c/d) only."
        )
        response = call_llm(prompt)
        predicted_letter = re.sub(r'[^a-d]', '', response.strip().lower())[:1]
        correct_letter = re.sub(r'[^a-d]', '', ep["answer"].strip().lower())[:1]
        is_correct = predicted_letter == correct_letter
        correct += int(is_correct)
        print(f"[{ep['episode_id']}] Predicted: {response.strip()} | Correct: {ep['answer']} | {'✅' if is_correct else '❌'}")
        time.sleep(1.5)
    print(f"\n--- NO MEMORY RESULTS ---")
    print(f"Accuracy: {correct}/{len(episodes)} = {correct/len(episodes):.2%}")

if __name__ == "__main__":
    print("Loading episodes from dev_10.jsonl...")
    episodes = []
    with open("data/dev_10.jsonl") as f:
        for line in f:
            episodes.append(json.loads(line))
    print(f"Loaded {len(episodes)} episodes.\n")
    run_no_memory(episodes)
