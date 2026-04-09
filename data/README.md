# PersonaMem Data Pipeline & Integrity Report

## Overview
This directory contains the scripts and processed data for the **Controlled Persistent Memory (CPM) for Long-Horizon Assistants** project. The pipeline transforms raw PersonaMem data into a standardized JSONL format for retrieval-augmented generation (RAG) and memory evaluation.

## File Structure
- `dev_latest.jsonl`: The standardized active dataset for team-wide evaluation.
- `build_episode.py`: The "factory" script that slices history and constructs episodes.
- `validate_episodes.py`: Structural validator ensuring schema and type consistency.
- `check_leakage.py`: Scientific audit script to prevent "future" data from leaking into history.
- `load_data.py`: Helper script for loading raw CSV and JSON sources.

## Complete Dataset Mapping (18 Keys)

| # | JSON Key | Source / Logic | Description |
| :--- | :--- | :--- | :--- |
| 1 | episode_id | Internal Counter | Unique integer ID for tracking specific test cases. |
| 2 | persona_id | persona_id | Links the episode to a specific character profile. |
| 3 | question_id | question_id | Unique identifier from the original PersonaMem dataset. |
| 4 | shared_context_id | shared_context_id | Reference ID for the full conversation thread. |
| 5 | question | user_question_or_message | The actual query the AI model must answer. |
| 6 | options | all_options | String containing the A/B/C/D multiple-choice options. |
| 7 | answer | correct_answer | The ground truth label (A, B, C, or D). |
| 8 | sliced_context | full_history[:end_idx-1] | The history as a labeled string (e.g., User: ... Assistant: ...). |
| 9 | turns | Structured List | A list of dictionaries (turn_index, role, content). |
| 10 | question_type | question_type | Categorization of the query (e.g., Reasoning vs. Recall). |
| 11 | topic | topic | The thematic subject of the conversation. |
| 12 | context_length_in_tokens | context_length_in_tokens | Total token count of the history provided to the model. |
| 13 | context_length_in_letters | context_length_in_letters | Total character count of the history string. |
| 14 | distance_to_ref_in_blocks | distance_to_ref_in_blocks | Number of context blocks between the answer and the query. |
| 15 | distance_to_ref_in_tokens | distance_to_ref_in_tokens | Number of tokens between the answer mention and the query. |
| 16 | num_irrelevant_tokens | num_irrelevant_tokens | Amount of "noise" tokens present in the conversation history. |
| 17 | distance_to_ref_proportion_in_context | Derived Proportion | Relative position of the answer in the history (0.0 to 1.0). |
| 18 | end_index_in_shared_context | end_index_in_shared_context | Original index of the turn where the question occurred. |

## Integrity & Slicing Logic
### The "Scientific Firewall"
To ensure a valid evaluation, a **-1 turn buffer** is applied to all episodes:
- **Logic:** `full_history[:max(0, end_idx - 1)]`
- **Reasoning:** In the raw dataset, the `end_index` typically points to the turn containing the question. By subtracting 1, we ensure the history ends strictly *before* the question is asked, preventing the AI from "reading ahead" or seeing the question in its own memory.

### Audit Results (Day 3 Milestone)
- **Status:** **PASSED**
- **Date:** March 30, 2026
- **Verification:** Automated check_leakage.py and manual spot checks confirmed zero literal string leakage of the question text into the sliced_context.

SCENARIO BREAKDOWN VARIABLES: 

1. topic

Simple Explanation: The subject of the conversation (e.g., Travel, Finance, or Music).

Why we chose this: To identify if the memory system has "blind spots" in specific domains. It reveals if the model is more reliable when remembering financial details versus creative hobbies.

2. question_type

Simple Explanation: Whether the answer requires simple fact retrieval (Recall) or connecting multiple pieces of information (Reasoning).

Why we chose this: To measure the "intelligence" of the retrieval process. It distinguishes between a system that can only repeat text and one that truly understands complex logic.

3. context_length_in_tokens

Simple Explanation: The total word count of the conversation history provided to the AI.

Why we chose this: Every AI has a capacity limit. This allows the team to pinpoint the exact "breaking point" where the history becomes too large for the system to process accurately.

4. distance_to_ref_in_tokens

Simple Explanation: How many words have passed between the mention of the answer and the user's question.

Why we chose this: This is the "Needle in a Haystack" test. It determines the effective range of the memory—showing if it is harder to find a fact from 5,000 words ago versus 500 words ago.

5. distance_to_ref_in_blocks

Simple Explanation: The number of back-and-forth turns between the answer and the question.

Why we chose this: Sometimes the number of interactions causes more memory decay than the word count. This identifies if frequent topic changes during the chat overwhelm the system.

6. distance_to_ref_proportion_in_context

Simple Explanation: The relative location of the answer in the file (0.0 = Start, 0.5 = Middle, 1.0 = End).

Why we chose this: To detect "Positional Bias." It mathematically proves if the system suffers from "Lost-in-the-Middle" syndrome, where it forgets information placed in the center of a long chat.

7. num_irrelevant_tokens

Simple Explanation: The amount of "small talk" or filler text that is unrelated to the answer.

Why we chose this: To measure "Noise Robustness." It tests the system's focus and its ability to ignore conversational distractions to find the one relevant fact.

8. end_index_in_shared_context

Simple Explanation: The total "age" of the conversation (e.g., is this Turn 10 or Turn 200?).

Why we chose this: To test for "Relationship Maturity." It determines if the memory system remains sharp and reliable over the course of a long-term, multi-day history.

9. persona_id

Simple Explanation: The specific identity and profile of the user (Gender, Race, Job, etc.).

Why we chose this: This is the "Fairness Audit." It ensures the system provides an equal quality of service to all users, regardless of their demographic background or profession.

10. context_length_in_letters

Simple Explanation: The literal character count of the conversation history.

Why we chose this: To track the "Physical Scale" of the data. This is used to compare database efficiency and processing speed against the token-based performance of the AI.