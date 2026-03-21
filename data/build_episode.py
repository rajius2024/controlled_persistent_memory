from load_data import load_questions, load_shared_contexts

def build_single_episode():
    questions_df = load_questions()
    contexts_dict = load_shared_contexts()
    
    # Select the first test case
    first_row = questions_df.iloc[0]
    
    # Use 'shared_context_id' to link to the JSONL data
    context_id = first_row.get("shared_context_id")
    end_index = int(first_row.get("end_index_in_shared_context"))
    
    full_context = contexts_dict.get(context_id)
    
    if full_context is None:
        raise ValueError(f"Could not find context for ID: {context_id}")
        
    # Perform the slice
    context_before_query = full_context[:end_index]
    
    episode = {
        "question_id": first_row.get("question_id"),
        "question": first_row.get("question"),
        "choices": first_row.get("all_options"),
        "answer": first_row.get("correct_answer"),
        "sliced_context": context_before_query
    }
    
    return episode

if __name__ == "__main__":
    try:
        episode = build_single_episode()
        print("\n[SUCCESS] Episode Built!")
        print(f"QUESTION: {episode['question']}")
        print(f"CORRECT ANSWER: {episode['answer']}")
        print("\n--- START OF CHAT ---")
        print(episode['sliced_context'][:500] + "...")
    except Exception as e:
        print(f"[ERROR] {e}")