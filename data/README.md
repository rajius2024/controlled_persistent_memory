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