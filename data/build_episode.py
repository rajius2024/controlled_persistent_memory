import json
import os
from data.load_data import load_questions, load_shared_contexts

# Use the same trick to ensure it saves exactly where you want it
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def build_episodes(num_to_build=None):
    questions_df = load_questions()
    contexts_dict = load_shared_contexts()

    # DYNAMIC LOGIC: Use provided number or the length of the dataframe
    total_available = len(questions_df)
    limit = num_to_build if num_to_build is not None else total_available
    
    # Ensure we don't try to build more than we have
    limit = min(limit, total_available)
    
    output_file = os.path.join(BASE_DIR, "dev_latest.jsonl")
    
    with open(output_file, "w", encoding='utf-8') as f:
        actual_count = 0
        for i in range(limit):
            row = questions_df.iloc[i]
            
            ctx_id = str(row.get("shared_context_id"))
            end_idx = int(row.get("end_index_in_shared_context"))
            
            # 1. Pull the full list of messages
            full_history_list = contexts_dict.get(ctx_id, [])
            
            # 2. Slice the LIST of messages (Perfect precision)
            sliced_history_list = full_history_list[:max(0, end_idx - 1)]

            structured_turns = []
            formatted_string = ""
            
            for idx, turn in enumerate(sliced_history_list):
                role = turn.get("role", "Unknown").capitalize()
                content = turn.get("content", "")
                
                # --- Prefix Cleaning ---
                prefixes = [f"{role}: ", f"{role.lower()}: ", "System: ", "system: "]
                clean_text = content
                for p in prefixes:
                    if clean_text.startswith(p):
                        clean_text = clean_text[len(p):]
                        break
                
                clean_text = clean_text.strip()
                
                # A. Add to Structured List
                structured_turns.append({
                    "turn_index": idx,
                    "role": role.lower(),
                    "content": clean_text
                })

                # B. Add to Formatted String
                formatted_string += f"{role}: {clean_text}\n"

            # --- LEAKAGE SAFETY GATE ---
            # If the question is already in the context, skip this episode.
            question_text = str(row.get("user_question_or_message")).lower().strip()
            if question_text in formatted_string.lower():
                print(f"⚠️ Skipping Episode {i}: Leakage detected.")
                continue 
            
            # --- The 100% Complete Dictionary ---
            episode = {
                "episode_id": int(i),
                "persona_id": str(row.get("persona_id")),
                "question_id": str(row.get("question_id")),
                "shared_context_id": ctx_id,
                "question": str(row.get("user_question_or_message")),
                "options": str(row.get("all_options")), 
                "answer": str(row.get("correct_answer")),
                "sliced_context": formatted_string.strip(), 
                "turns": structured_turns,               
                "question_type": str(row.get("question_type")),
                "topic": str(row.get("topic")),
                "context_length_in_tokens": int(row.get("context_length_in_tokens", 0)),
                "context_length_in_letters": int(row.get("context_length_in_letters", 0)),
                "distance_to_ref_in_blocks": int(row.get("distance_to_ref_in_blocks", 0)),
                "distance_to_ref_in_tokens": int(row.get("distance_to_ref_in_tokens", 0)),
                "num_irrelevant_tokens": int(row.get("num_irrelevant_tokens", 0)),
                "distance_to_ref_proportion_in_context": float(str(row.get("distance_to_ref_proportion_in_context", "0.0")).replace("%", "")) / 100,   
                "end_index_in_shared_context": end_idx
            }
            
            f.write(json.dumps(episode) + "\n")
            actual_count += 1
            
    print(f"Success: Created {actual_count} clean episodes in 'dev_latest.jsonl' (Skipped {limit - actual_count} leaks)")

if __name__ == "__main__":
    build_episodes()