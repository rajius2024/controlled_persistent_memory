import pandas as pd
import json
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "questions_32k.csv")
JSONL_PATH = os.path.join(BASE_DIR, "shared_contexts_32k.jsonl")

def load_questions():
    print(f"Loading questions from {CSV_PATH}...")
    df = pd.read_csv(CSV_PATH)
    return df

def load_shared_contexts():
    print(f"Loading shared contexts from {JSONL_PATH}...")
    contexts_dict = {}
    
    with open(JSONL_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            record = json.loads(line)
            
            context_id = list(record.keys())[0]
            messages = record[context_id]
            
            # KEEP AS A LIST for correct slicing later!
            contexts_dict[context_id] = messages 
                
    return contexts_dict

if __name__ == "__main__":
    q = load_questions()
    c = load_shared_contexts()
    print(f"\n--- SUCCESS ---")
    print(f"Loaded {len(q)} questions and {len(c)} contexts.")