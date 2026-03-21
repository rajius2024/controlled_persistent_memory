import pandas as pd
import json
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "questions_32k.csv")
JSONL_PATH = os.path.join(BASE_DIR, "shared_contexts_32k.jsonl")

def load_questions():
    print(f"Loading questions from {CSV_PATH}...")
    df = pd.read_csv(CSV_PATH)
    
    # We rename 'user_question_or_message' to 'question' for simplicity
    if 'user_question_or_message' in df.columns:
        df = df.rename(columns={'user_question_or_message': 'question'})
        
    return df

def load_shared_contexts():
    print(f"Loading shared contexts from {JSONL_PATH}...")
    contexts_dict = {}
    
    with open(JSONL_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            record = json.loads(line)
            
            # Each line is a dict like {"hash_id": [messages]}
            # We extract that hash_id to match the 'shared_context_id' in your CSV
            context_id = list(record.keys())[0]
            messages = record[context_id]

            content = ""
            for turn in messages:
                role = turn.get("role", "Unknown").capitalize()
                text = turn.get("content", "")
                content += f"{role}: {text}\n"

            contexts_dict[context_id] = content
                
    return contexts_dict

if __name__ == "__main__":
    q = load_questions()
    c = load_shared_contexts()
    print(f"\n--- SUCCESS ---")
    print(f"Loaded {len(q)} questions and {len(c)} contexts.")