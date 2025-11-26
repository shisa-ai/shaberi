import argparse
import os
import json
from datetime import datetime, timezone
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

from datasets import Dataset, load_dataset

from evaluation_datasets_config import (
    EVAL_MODEL_CONFIGS,
    get_ans_path,
    ensure_id_column,
    filter_excluded_questions_by_id,
)
import llm_functions
from llm_functions import get_answerer


def load_model_dataset(evaluation_dataset_name: str) -> Dataset:
    eval_config = EVAL_MODEL_CONFIGS.get(evaluation_dataset_name, None)

    if eval_config is None:
        raise ValueError(f'モデル名「{evaluation_dataset_name}」は対応しておりません。引数の"--eval_dataset_name"は{list(EVAL_MODEL_CONFIGS.keys())}から選択してください。')

    # Load dataset
    dataset = load_dataset(evaluation_dataset_name)

    # Get specified split
    split_name = eval_config.get("split_name", "test")
    split_dataset = dataset[split_name]

    # Map question column to standardised column name
    q_col = eval_config.get("question_column")
    if q_col != "Question":
        split_dataset = split_dataset.rename_column(q_col, "Question")

    # Ensure each row has a stable integer id and apply any dataset-level
    # exclusions (e.g. specific Tengu questions that should be skipped).
    split_dataset = ensure_id_column(split_dataset, id_column="id")
    split_dataset = filter_excluded_questions_by_id(evaluation_dataset_name, split_dataset, id_column="id")

    return split_dataset

def process_question(question: str, model_name: str, answer_function):
    """Worker function to process a single question."""
    try:
        model_answer = answer_function(question, model_name)
        return question, model_answer, None
    except Exception as e:
        return question, f"UNEXPECTED_ERROR: {str(e)}", str(e)

def run_generate_with_checkpoint(model_name: str,
                                 eval_dataset_name: str = "all",
                                 num_proc: int = 16,
                                 rerun: bool = False,
                                 error_retry_limit: int = 3,
                                 force_retry_permanent: bool = False):
    
    eval_dataset_names = list(EVAL_MODEL_CONFIGS.keys()) if eval_dataset_name == "all" else [eval_dataset_name]
    
    for dataset_name in eval_dataset_names:
        print(f"Processing {dataset_name} with {model_name}")
        
        # 1. Load test dataset
        dataset = load_model_dataset(dataset_name)
        model_answer_path = get_ans_path(dataset_name, model_name)
        
        # 2. Load existing answers if available and not rerunning
        existing_answers = {}
        existing_meta = {}
        if os.path.exists(model_answer_path) and not rerun:
            try:
                total_lines = 0
                errors = 0
                incomplete = 0
                with open(model_answer_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        if not line.strip():
                            continue
                        total_lines += 1
                        item = json.loads(line)
                        q = item.get('Question')
                        ans = item.get('ModelAnswer')
                        # Missing or empty answers are incomplete
                        if q is None or (ans is None) or (isinstance(ans, str) and ans.strip() == ""):
                            incomplete += 1
                            continue
                        # Errors are kept (so they impact scoring) but counted separately
                        if isinstance(ans, str) and ans.startswith(('ERROR', 'PERMANENT_ERROR', 'UNEXPECTED_ERROR')):
                            errors += 1
                        existing_answers[q] = ans
                        # Load metadata if present; default if absent
                        existing_meta[q] = {
                            'ErrorCount': int(item.get('ErrorCount', 0) or 0),
                            'ErrorType': item.get('ErrorType'),
                            'UpdatedAt': item.get('UpdatedAt')
                        }
                print(f"Loaded {len(existing_answers)} answers ({errors} errors, {incomplete} incomplete of {total_lines}).")
            except Exception as e:
                print(f"Failed to load existing answers: {e}")
                existing_answers = {}
                existing_meta = {}
        
        # 3. Generate answers with checkpointing
        answer_function = get_answerer(model_name)

        # Pre-index items by question for reliable writes
        item_by_q = {item['Question']: item for item in dataset}

        # Track answers by question; start with existing answers (including errors) unless rerun
        answers_by_q = {} if rerun else dict(existing_answers)
        meta_by_q = {} if rerun else dict(existing_meta)

        # Decide which questions to process
        def classify_error(ans: str | None) -> str | None:
            if not isinstance(ans, str):
                return None
            if ans.startswith('PERMANENT_ERROR'):
                return 'PERMANENT_ERROR'
            if ans.startswith('ERROR'):
                return 'ERROR'
            if ans.startswith('UNEXPECTED_ERROR'):
                return 'UNEXPECTED_ERROR'
            return None

        to_run = set()
        done_count = 0
        permanent_skipped = 0
        retry_error_count = 0
        for q in item_by_q.keys():
            if rerun:
                to_run.add(q)
                continue
            ans = existing_answers.get(q)
            if ans is None or (isinstance(ans, str) and ans.strip() == ""):
                to_run.add(q)
                continue
            etype = classify_error(ans)
            if etype == 'PERMANENT_ERROR' and not force_retry_permanent:
                done_count += 1
                permanent_skipped += 1
                continue
            if etype in ('ERROR', 'UNEXPECTED_ERROR'):
                attempts = int(meta_by_q.get(q, {}).get('ErrorCount', 0) or 0)
                if attempts < error_retry_limit:
                    to_run.add(q)
                    retry_error_count += 1
                else:
                    done_count += 1
                continue
            # Valid answer
            done_count += 1

        # Calculate progress and set up progress bar
        total_items = len(dataset)
        cached_count = 0 if rerun else done_count
        pbar_desc = f"Generating answers for {dataset_name} (cached={cached_count}, retry_errors={retry_error_count}, permanent={permanent_skipped})"
        pbar = tqdm(total=total_items, initial=cached_count, desc=pbar_desc)

        generated_since_checkpoint = 0

        def now_iso():
            return datetime.now(timezone.utc).isoformat()

        def write_checkpoint_atomic(path: str):
            tmp_path = path + ".tmp"
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(tmp_path, 'w', encoding='utf-8') as f:
                # Write only items that have answers (cached or new)
                for q in item_by_q.keys():
                    if q in answers_by_q:
                        base = dict(item_by_q[q])
                        base['ModelAnswer'] = answers_by_q[q]
                        meta = meta_by_q.get(q)
                        if meta:
                            # Only add metadata fields; judge ignores extra columns
                            base['ErrorCount'] = int(meta.get('ErrorCount', 0) or 0)
                            if meta.get('ErrorType') is not None:
                                base['ErrorType'] = meta.get('ErrorType')
                            if meta.get('UpdatedAt') is not None:
                                base['UpdatedAt'] = meta.get('UpdatedAt')
                        json.dump(base, f, ensure_ascii=False)
                        f.write('\n')
            # Atomic replace to avoid truncation on interruption
            os.replace(tmp_path, path)

        # Process questions using ThreadPoolExecutor
        questions_to_process = [item['Question'] for item in dataset if item['Question'] in to_run]

        with ThreadPoolExecutor(max_workers=num_proc) as executor:
            # Submit all questions to the thread pool
            future_to_question = {
                executor.submit(process_question, question, model_name, answer_function): question
                for question in questions_to_process
            }

            try:
                # Process completed futures as they finish
                for future in as_completed(future_to_question):
                    question = future_to_question[future]

                    try:
                        question_result, model_answer, error = future.result()

                        # Store the answer
                        answers_by_q[question] = model_answer

                        # Update metadata
                        prev = meta_by_q.get(question, {'ErrorCount': 0, 'ErrorType': None, 'UpdatedAt': None})
                        etype = classify_error(model_answer)
                        if etype is None:
                            # Successful answer; preserve prior error count (no increment)
                            meta_by_q[question] = {'ErrorCount': int(prev.get('ErrorCount', 0) or 0), 'ErrorType': None, 'UpdatedAt': now_iso()}
                        else:
                            meta_by_q[question] = {'ErrorCount': int(prev.get('ErrorCount', 0) or 0) + 1, 'ErrorType': etype, 'UpdatedAt': now_iso()}

                        generated_since_checkpoint += 1

                        # Checkpoint: Save progress every 10 new generations or on error answers
                        if (generated_since_checkpoint % 10 == 0) or (isinstance(model_answer, str) and model_answer.startswith(('ERROR', 'PERMANENT_ERROR', 'UNEXPECTED_ERROR'))):
                            # Save intermediate results atomically (merge cached + new)
                            write_checkpoint_atomic(model_answer_path)

                            if isinstance(model_answer, str) and model_answer.startswith(('ERROR', 'PERMANENT_ERROR', 'UNEXPECTED_ERROR')):
                                print(f"Error on question, checkpointed progress")

                        # Update progress bar
                        pbar.update(1)

                    except Exception as e:
                        print(f"Unexpected error processing question result: {e}")
                        # Still add the question with an error message
                        err_msg = f"UNEXPECTED_ERROR: {str(e)}"
                        answers_by_q[question] = err_msg
                        prev = meta_by_q.get(question, {'ErrorCount': 0, 'ErrorType': None, 'UpdatedAt': None})
                        meta_by_q[question] = {'ErrorCount': int(prev.get('ErrorCount', 0) or 0) + 1, 'ErrorType': 'UNEXPECTED_ERROR', 'UpdatedAt': now_iso()}
                        generated_since_checkpoint += 1
                        pbar.update(1)

            except KeyboardInterrupt:
                print("\nInterrupted by user. Saving progress...")
                # Cancel remaining futures
                for future in future_to_question:
                    future.cancel()
                # Save current progress atomically before exiting (merge cached + new)
                write_checkpoint_atomic(model_answer_path)
                print(f"Progress saved to {model_answer_path}")
                raise
        
        # 4. Final save (atomic)
        write_checkpoint_atomic(model_answer_path)
        pbar.close()
        
        print(f"Completed {dataset_name}, saved {len(answers_by_q)} answers to {model_answer_path}")

def main():
    parser = argparse.ArgumentParser(description='Generate model answers with checkpointing')

    parser.add_argument('-m', '--model_name', type=str, required=True)
    parser.add_argument('-d', '--eval_dataset_name', type=str, default='all')
    parser.add_argument('-n', '--num_proc', type=int, default=8)
    parser.add_argument('-fp', '--frequency_penalty', type=float, default=1.0)
    parser.add_argument('--rerun', action='store_true', 
                       help='Regenerate all answers, ignoring existing ones')
    parser.add_argument('--error-retry-limit', type=int, default=3, help='Max attempts for transient errors (ERROR/UNEXPECTED_ERROR)')
    parser.add_argument('--force-retry-permanent', action='store_true', help='Retry PERMANENT_ERROR entries as well')

    args = parser.parse_args()

    # hack
    llm_functions.fp = args.frequency_penalty

    run_generate_with_checkpoint(
        args.model_name,
        args.eval_dataset_name,
        args.num_proc,
        args.rerun,
        args.error_retry_limit,
        args.force_retry_permanent,
    )
    
if __name__ == '__main__':
    main()
