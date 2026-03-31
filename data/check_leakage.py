import json
import os

# Identify the file location dynamically within the /data folder
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Matches the standard name used in build_episode.py and validate_episodes.py
DATA_FILE = os.path.join(BASE_DIR, "dev_latest.jsonl")

def run_leakage_audit():
    """
    Checks every episode in the latest build to ensure the question text 
    does not appear in the conversation history (leakage).
    """
    if not os.path.exists(DATA_FILE):
        print(f"❌ Error: {DATA_FILE} not found. Run build_episode.py first.")
        return

    # 1. Load the dataset
    episodes = []
    with open(DATA_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                episodes.append(json.loads(line))

    total_episodes = len(episodes)
    leaks_found = 0
    
    print("="*60)
    print(f"📊 DATA LEAKAGE AUDIT: {total_episodes} EPISODES")
    print("="*60)

    for ep in episodes:
        episode_id = ep.get("episode_id")
        query = ep.get("question", "").strip()
        context = ep.get("sliced_context", "")

        # Case-insensitive check: Does the question exist in the history?
        if query.lower() in context.lower():
            leaks_found += 1
            print(f"[ID: {episode_id}] ❌ LEAK DETECTED")
        else:
            # Print a dot for each clean episode to show progress without clutter
            print(".", end="", flush=True)

    # 2. Final Summary Table
    print("\n" + "="*60)
    print(f"FINAL AUDIT RESULTS for {os.path.basename(DATA_FILE)}")
    print(f"Total Checked: {total_episodes}")
    print(f"Leakage Free:  {total_episodes - leaks_found}")
    print(f"Status:        {'PASSED' if leaks_found == 0 else 'FAILED'}")
    print("="*60)

    if leaks_found > 0:
        print(f"\nWARNING: {leaks_found} cases of leakage found. Check build_episode.py slicing.")

if __name__ == "__main__":
    run_leakage_audit()