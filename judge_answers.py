import argparse
import os
import re

from datasets import load_dataset

reasoning_pattern = re.compile(r'<(think|thinking|reason)>.*?</(think|thinking|reason)>', re.DOTALL | re.IGNORECASE)

from evaluation_datasets_config import (
    EVAL_MODEL_CONFIGS,
    get_ans_path,
    ensure_id_column,
    filter_excluded_questions_by_id,
)


def evaluate(model_name: str, eval_dataset_name: str, evaluation_model: str, num_proc: int, rerun: bool):

    model_answer_path = get_ans_path(eval_dataset_name, model_name)
    ans_dataset = load_dataset('json', data_files=model_answer_path, split="train")

    eval_config = EVAL_MODEL_CONFIGS.get(eval_dataset_name, None)

    if eval_config is None:
        raise ValueError(f'モデル名「{eval_dataset_name}」は対応しておりません。引数の"--eval_dataset_name"は{list(EVAL_MODEL_CONFIGS.keys())}から選択してください。')

    eval_fn = eval_config["evaluator_function"]

    # First, normalize and filter the answer dataset:
    # - Ensure a stable integer id per row
    # - Drop any dataset-level excluded questions (e.g. specific Tengu items)
    # - Remove reasoning/thinking segments from answers
    ans_dataset = ensure_id_column(ans_dataset, id_column="id")
    ans_dataset = filter_excluded_questions_by_id(eval_dataset_name, ans_dataset, id_column="id")
    ans_dataset = ans_dataset.map(
        lambda x: {"ModelAnswer": reasoning_pattern.sub("", x.get("ModelAnswer") or "")},
        num_proc=num_proc,
        load_from_cache_file=True,
    )

    # Then, run the judge to compute scores (and optionally capture the
    # judge's full textual evaluation for future inspection).
    #
    # We want caching for speed on normal runs, but on --rerun we force a
    # fresh judging pass by disabling cache loading for this map only.
    def _apply_judge(row: dict) -> dict:
        """
        Apply the evaluator function and normalize its output.

        - If the evaluator returns a dict (e.g., {"score": 7, "judge_output": "..."}),
          pass it through so additional fields become new dataset columns.
        - If it returns a bare score (legacy behaviour), wrap it into {"score": value}.
        """
        result = eval_fn(row, evaluation_model)
        if isinstance(result, dict):
            return result
        return {"score": result}

    ans_dataset = ans_dataset.map(
        _apply_judge,
        num_proc=num_proc,
        load_from_cache_file=not rerun,
    )

    output_dir = os.path.join(".", "data", "judgements", "judge_" + evaluation_model.replace("/", "__"), eval_dataset_name.replace("/", "__"))
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, model_name.replace("/", "__") + ".json")

    if os.path.exists(output_path) and not rerun:
        print(f"Skipping {model_name} on {eval_dataset_name} (existing judgement found at {output_path}; use --rerun to overwrite)")
        return

    ans_dataset.to_json(output_path)


def run_judgement(model_name: str, eval_dataset_name: str = "all", evaluation_model: str = "gpt-4.1-2025-04-14", num_proc: int = 8, rerun: bool = False):
    eval_dataset_names = EVAL_MODEL_CONFIGS.keys() if eval_dataset_name == "all" else [eval_dataset_name]

    for eval_dataset_name in eval_dataset_names:
        print(f"Judging {model_name} on {eval_dataset_name} using {evaluation_model} ({num_proc} proc)")
        evaluate(model_name, eval_dataset_name, evaluation_model, num_proc, rerun)
        
def main():
    parser = argparse.ArgumentParser(description='Judge model answers with LLM as judge')

    parser.add_argument('-m', '--model_name', type=str, required=True)
    parser.add_argument('-d', '--eval_dataset_name', type=str, default='all')
    parser.add_argument(
        '-e', '--evaluation_model', '--judge-model',
        dest='evaluation_model',
        type=str,
        # default='gpt-4.1-2025-04-14'
        default='gpt-5.1-2025-11-13'
    )
    parser.add_argument('--judge-base-url', type=str, default=None)
    parser.add_argument('--judge-api-key', type=str, default=None)
    parser.add_argument('--judge-api-key-env', type=str, default=None)
    parser.add_argument('--gemini-judge', action='store_true', help='Use native Gemini API instead of OpenAI-compatible endpoint')
    parser.add_argument('-n', '--num_proc', type=int, default=20)
    parser.add_argument('--rerun', action='store_true', help='Rejudge even if an existing judgement file is present')

    args = parser.parse_args()

    is_gemini_model = args.evaluation_model.lower().startswith("gemini")

    # Apply Gemini-friendly defaults when using Gemini judges
    judge_base_url = args.judge_base_url
    judge_api_key_env = args.judge_api_key_env
    if is_gemini_model:
        if judge_base_url is None:
            judge_base_url = "https://generativelanguage.googleapis.com/v1beta/openai/"
        if judge_api_key_env is None and args.judge_api_key is None:
            judge_api_key_env = "GEMINI_API_KEY"

    # Allow overriding judge credentials/endpoint without touching global env
    judge_api_key = None
    if judge_api_key_env:
        judge_api_key = os.environ.get(judge_api_key_env)
    if args.judge_api_key:
        judge_api_key = args.judge_api_key

    if judge_api_key:
        os.environ["OPENAI_API_KEY"] = judge_api_key
    if judge_base_url:
        os.environ["OPENAI_BASE_URL"] = judge_base_url
    if args.gemini_judge:
        os.environ["GEMINI_NATIVE"] = "1"

    run_judgement(args.model_name, args.eval_dataset_name, args.evaluation_model, args.num_proc, args.rerun)
    
if __name__ == '__main__':
    main()
