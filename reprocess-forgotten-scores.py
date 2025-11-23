import argparse
import json
import os

from rich.console import Console

from evaluation_datasets_config import EVAL_MODEL_CONFIGS


# Disable Rich's automatic syntax highlighting so that numbers inside
# model names / file paths (e.g. "llama3.3-70b-v2.1") are not colored.
# Only explicit markup like [bold red]FAILED:[/bold red] will be styled.
console = Console(highlight=False)


def _resolve_judge_dir(evaluation_model: str) -> str:
    """
    Map a judge model name (e.g. 'gpt-5.1-2025-11-13') to the directory
    where its judgements are stored.
    """
    judge_dir_name = "judge_" + evaluation_model.replace("/", "__")
    return os.path.join(".", "data", "judgements", judge_dir_name)


def _reprocess_file(
    file_path: str,
    dataset_name: str,
    evaluation_model: str,
    dry_run: bool,
) -> tuple[int, int, int]:
    """
    Reprocess a single JSONL judgement file.

    Returns:
        (total_rows, missing_before, fixed_now)
    """
    eval_config = EVAL_MODEL_CONFIGS.get(dataset_name)
    if eval_config is None:
        raise ValueError(f"Unknown dataset_name '{dataset_name}'. Known datasets: {list(EVAL_MODEL_CONFIGS.keys())}")

    eval_fn = eval_config["evaluator_function"]

    total_rows = 0
    missing_before = 0
    fixed_now = 0

    tmp_path = file_path + ".tmp"

    with open(file_path, "r", encoding="utf-8") as src, open(tmp_path, "w", encoding="utf-8") as dst:
        for line in src:
            if not line.strip():
                continue

            record = json.loads(line)
            total_rows += 1
            score = record.get("score", None)

            if score is None:
                missing_before += 1
                try:
                    new_score = eval_fn(record, evaluation_model)
                except Exception as e:
                    console.print(
                        f"[bold red]FAILED:[/bold red] error while re-evaluating a sample in {file_path}: {e}",
                        highlight=False,
                    )
                    new_score = None

                record["score"] = new_score
                if new_score is not None:
                    fixed_now += 1

            dst.write(json.dumps(record, ensure_ascii=False) + "\n")

    if dry_run:
        os.remove(tmp_path)
    else:
        os.replace(tmp_path, file_path)

    return total_rows, missing_before, fixed_now


def main():
    parser = argparse.ArgumentParser(
        description="Reprocess judgements with missing scores using the updated LLM judge fallback."
    )
    # NOTE: -e / --evaluation_model controls which judge's results
    # directory we scan under data/judgements/judge_<evaluation_model>.
    parser.add_argument(
        "-e",
        "--evaluation_model",
        "--judge-model",
        dest="evaluation_model",
        type=str,
        default="gpt-5.1-2025-11-13",
        help=(
            "Judge model name whose results to reprocess "
            "(maps to data/judgements/judge_<evaluation_model>). "
            "Default: gpt-5.1-2025-11-13"
        ),
    )
    parser.add_argument(
        "-d",
        "--eval_dataset_name",
        type=str,
        default="all",
        help="Dataset name to reprocess (e.g. 'lightblue/tengu_bench', 'elyza/ELYZA-tasks-100') or 'all'.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Scan and report counts but do not modify any files.",
    )

    # Optional: mirror judge_answers.py environment overrides so the same
    # endpoint/API-key setup can be reused for reprocessing.
    parser.add_argument("--judge-base-url", type=str, default=None)
    parser.add_argument("--judge-api-key", type=str, default=None)
    parser.add_argument("--judge-api-key-env", type=str, default=None)
    parser.add_argument(
        "--gemini-judge",
        action="store_true",
        help="Use native Gemini API instead of OpenAI-compatible endpoint",
    )

    args = parser.parse_args()

    evaluation_model = args.evaluation_model

    judge_base_url = args.judge_base_url
    judge_api_key_env = args.judge_api_key_env

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

    judge_dir = _resolve_judge_dir(evaluation_model)

    if not os.path.isdir(judge_dir):
        console.print(
            f"[bold red]FAILED:[/bold red] judge directory not found for model '{evaluation_model}': {judge_dir}",
            highlight=False,
        )
        raise SystemExit(1)

    if args.eval_dataset_name == "all":
        dataset_names = list(EVAL_MODEL_CONFIGS.keys())
    else:
        dataset_names = [args.eval_dataset_name]

    total_rows_overall = 0
    total_missing = 0
    total_fixed = 0

    for dataset_name in dataset_names:
        dataset_dir_name = dataset_name.replace("/", "__")
        dataset_dir = os.path.join(judge_dir, dataset_dir_name)
        if not os.path.isdir(dataset_dir):
            continue

        console.print(
            f"Reprocessing dataset '{dataset_name}' for judge '{evaluation_model}' in {dataset_dir}...",
            highlight=False,
        )

        for fname in sorted(os.listdir(dataset_dir)):
            if not fname.endswith(".json"):
                continue
            fpath = os.path.join(dataset_dir, fname)

            file_rows, missing_before, fixed_now = _reprocess_file(
                fpath,
                dataset_name,
                evaluation_model,
                dry_run=args.dry_run,
            )

            if missing_before > 0:
                console.print(
                    f"  File [gold3]{fname}[/gold3]: "
                    f"rows = [teal]{file_rows}[/teal], "
                    f"missing before = [teal]{missing_before}[/teal], "
                    f"fixed now = [teal]{fixed_now}[/teal]"
                )

            if missing_before > fixed_now:
                # Some samples in this file still have no score even after
                # reprocessing; highlight this clearly in red so they stand out.
                remaining = missing_before - fixed_now
                console.print(
                    f"[bold red]FAILED:[/bold red] {fname} – {remaining} score(s) still missing after reprocess",
                    highlight=False,
                )

            total_rows_overall += file_rows
            total_missing += missing_before
            total_fixed += fixed_now

    mode = "DRY-RUN" if args.dry_run else "UPDATED"
    total_attempts = total_missing
    total_failures = max(total_attempts - total_fixed, 0)

    console.print(
        f"[{mode}] Total rows scanned: {total_rows_overall}, "
        f"rows needing reprocess: {total_attempts}, "
        f"rows successfully backfilled: {total_fixed}, "
        f"rows still missing score: {total_failures}",
        highlight=False,
    )

    if total_rows_overall > 0:
        reprocess_pct = (total_attempts / total_rows_overall) * 100.0
    else:
        reprocess_pct = 0.0

    if total_attempts > 0:
        success_rate = (total_fixed / total_attempts) * 100.0
        failure_rate = (total_failures / total_attempts) * 100.0
    else:
        success_rate = 0.0
        failure_rate = 0.0

    console.print(
        f"[{mode}] Reprocessed rows: {total_attempts} "
        f"({reprocess_pct:.2f}% of all rows). "
        f"Success rate: {success_rate:.2f}%, "
        f"failure rate: {failure_rate:.2f}%.",
        highlight=False,
    )


if __name__ == "__main__":
    main()
