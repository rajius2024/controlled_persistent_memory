import json
import os

# Identify the file location dynamically within the /data folder
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FILE_PATH = os.path.join(BASE_DIR, "dev_10.jsonl")

def validate():
    if not os.path.exists(FILE_PATH):
        print(f"Target file not found: {FILE_PATH}")
        return

    # THE STABLE CONTRACT: Defines keys and expected Python types
    schema = {
        "episode_id": int,
        "persona_id": str,
        "question_id": str,
        "shared_context_id": str,
        "question": str,
        "options": str,
        "answer": str,
        "turns": list,
        "sliced_context": str,
        "context_length_in_tokens": int,
        "end_index_in_shared_context": int,
        "distance_to_ref_proportion_in_context": float
    }

    print(f"🔍 Starting Deep Sanity Check on {os.path.basename(FILE_PATH)}...")

    with open(FILE_PATH, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            try:
                ep = json.loads(line)
                
                # 1. EXISTENCE & TYPE CONSISTENCY
                for key, expected_type in schema.items():
                    if key not in ep: 
                        raise KeyError(f"Missing required key: '{key}'")
                    if not isinstance(ep[key], expected_type): 
                        raise TypeError(f"Key '{key}' should be {expected_type}, but got {type(ep[key])}")

                # 2. CONTENT SANITY (No empty strings or lists)
                if not ep["question"].strip(): raise ValueError("Question is empty")
                if not ep["options"].strip(): raise ValueError("Options field is empty")
                if not ep["sliced_context"].strip(): raise ValueError("sliced_context string is empty")
                if not ep["turns"]: raise ValueError("Turn list is empty")

                # 3. SCIENTIFIC METRIC CHECK (Proportion must be 0.0 to 1.0)
                prop = ep["distance_to_ref_proportion_in_context"]
                if not (0.0 <= prop <= 1.0):
                    raise ValueError(f"Proportion {prop} is out of scientific bounds (0-1)")

                # 4. ORDERING & DATA LEAKAGE CHECKS
                cutoff = ep["end_index_in_shared_context"]
                if len(ep["turns"]) != cutoff:
                    raise ValueError(f"Cutoff mismatch: Has {len(ep['turns'])} turns, expected {cutoff}")

                for idx, turn in enumerate(ep["turns"]):
                    # Verify ordering is preserved
                    if turn.get("turn_index") != idx:
                        raise ValueError(f"Ordering broken at Turn {idx}. Found index {turn.get('turn_index')}")
                    
                    # Verify no future leakage (Index must be less than cutoff)
                    if idx >= cutoff:
                        raise ValueError(f"Future Leak: Turn {idx} exists beyond cutoff {cutoff}")
                    
                    # Verify turn content exists
                    if not str(turn.get("content", "")).strip():
                        raise ValueError(f"Turn {idx} has empty content")

                print(f"Episode {i}: Validated and Traceable.")

            except Exception as e:
                print(f"Episode {i} FAILED: {e}")
                return # Stop immediately if data is corrupted

    print("\n DATA IS READY")

if __name__ == "__main__":
    validate()