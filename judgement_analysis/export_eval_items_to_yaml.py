#!/usr/bin/env python3
"""
Export per-question review files for all main eval datasets.

For each dataset, this script creates YAML-compatible files (JSON syntax that
is also valid YAML) under:

  judgement_analysis/export/<dataset>/qXXX.yaml

Each file contains:
  - Question metadata (category, question, gold answer, rubric)
  - Placeholders for English translations / explanations
  - Sample model answers for three models:
      * 183 (shisa-ai__183-llama3.1-8b-v2.1o-dpo-7e8.json)
      * Shisa V2 405B (shisa-ai__shisa-v2-llama3.1-405b.json)
      * Gemini 3 Pro preview (gemini-3-pro-preview.json)
  - Judge output from GPT-5.1 (judge_gpt-5.1-2025-11-13), when available

These files are meant for manual rubric / instruction review and can be
extended later with additional judge runs or annotations.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Iterator, List, Optional

try:
    import yaml  # type: ignore[import]

    HAVE_YAML = True
except Exception:
    yaml = None  # type: ignore[assignment]
    HAVE_YAML = False


ROOT = Path(__file__).resolve().parents[1]

# Datasets we actively care about for re-review.
DATASETS: List[str] = [
    "elyza__ELYZA-tasks-100",
    "lightblue__tengu_bench",
    "shisa-ai__ja-mt-bench-1shot",
    "yuzuai__rakuda-questions",
]

# Primary judge whose outputs we will include as samples.
PRIMARY_JUDGE = "judge_gpt-5.1-2025-11-13"

# Model answer files to sample for each dataset.
# Keys are short identifiers we will expose in the YAML, values are filenames.
MODEL_FILES: Dict[str, str] = {
    "183": "shisa-ai__183-llama3.1-8b-v2.1o-dpo-7e8.json",
    "shisa-v2-405b": "shisa-ai__shisa-v2-llama3.1-405b.json",
    "gemini3-pro": "gemini-3-pro-preview.json",
}


def read_jsonl(path: Path) -> Iterator[dict]:
    """Yield JSON objects from a JSONL file."""
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def ensure_exists(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(str(path))


def export_dataset(dataset: str) -> None:
    """Export all questions for a single dataset into per-question YAML files."""
    dataset_dir = ROOT / "data" / "model_answers" / dataset
    if not dataset_dir.exists():
        print(f"[WARN] Dataset not found under model_answers: {dataset}")
        return

    out_dir = ROOT / "judgement_analysis" / "export" / dataset
    out_dir.mkdir(parents=True, exist_ok=True)

    # Prepare iterators for model answers and judge outputs
    model_iters: Dict[str, Iterator[dict]] = {}
    judge_iters: Dict[str, Optional[Iterator[dict]]] = {}

    for model_key, filename in MODEL_FILES.items():
        ma_path = dataset_dir / filename
        try:
            ensure_exists(ma_path)
        except FileNotFoundError:
            print(f"[WARN] Missing model_answers for {dataset}: {filename}, skipping dataset.")
            return
        model_iters[model_key] = read_jsonl(ma_path)

        judge_path = ROOT / "data" / "judgements" / PRIMARY_JUDGE / dataset / filename
        if judge_path.exists():
            judge_iters[model_key] = read_jsonl(judge_path)
        else:
            print(f"[WARN] Missing judgements for {dataset}: {PRIMARY_JUDGE}/{filename}")
            judge_iters[model_key] = None

    # Drive iteration from the 183 model answers, assuming consistent ordering.
    num_exported = 0
    for index, base_row in enumerate(model_iters["183"], start=1):
        # Base metadata (same across models)
        category = base_row.get("Category")
        question_text = base_row.get("Question")
        gold_answer = base_row.get("Answer")
        criteria = base_row.get("Criteria")

        # Default id is the 1-based index; we may override from judge rows.
        item_id: int = index

        samples: List[dict] = []
        # For each model, collect its answer and (optionally) judge info.
        for model_key, filename in MODEL_FILES.items():
            if model_key == "183":
                ans_row = base_row
            else:
                try:
                    ans_row = next(model_iters[model_key])
                except StopIteration:
                    print(f"[WARN] Early EOF in model_answers for {dataset} {filename}")
                    continue

            sample: dict = {
                "model_key": model_key,
                "model_file": filename,
                "model_answer": ans_row.get("ModelAnswer"),
                "judgements": [],
            }

            j_iter = judge_iters.get(model_key)
            if j_iter is not None:
                try:
                    j_row = next(j_iter)
                except StopIteration:
                    print(f"[WARN] Early EOF in judgements for {dataset} {filename}")
                    j_row = None
                if j_row is not None:
                    # Use the judge's id field if present.
                    j_id = j_row.get("id")
                    if isinstance(j_id, int):
                        item_id = j_id
                    sample["judgements"].append(
                        {
                            "judge_id": PRIMARY_JUDGE,
                            "score": j_row.get("score"),
                            "judge_output": j_row.get("judge_output"),
                            "error_count": j_row.get("ErrorCount"),
                            "updated_at": j_row.get("UpdatedAt"),
                        }
                    )

            samples.append(sample)

        # Build the review object
        review_obj = {
            "dataset": dataset,
            "id": int(item_id),
            "category": category,
            "question": {
                "original": question_text,
                # To be filled by reviewers for non-Japanese speakers.
                "translation_en": None,
            },
            "answer": {
                "gold_original": gold_answer,
                "gold_backtranslation_en": None,
            },
            "rubric": {
                "original": criteria,
                "explanation_en": None,
            },
            "samples": samples,
            "notes": "",
            "status": "unreviewed",
        }

        # Write as YAML when possible (more human-readable), otherwise JSON.
        out_path = out_dir / f"q{int(item_id):03d}.yaml"
        with out_path.open("w", encoding="utf-8") as out_f:
            if HAVE_YAML:
                yaml.safe_dump(
                    review_obj,
                    out_f,
                    allow_unicode=True,
                    sort_keys=False,
                )
            else:
                json.dump(review_obj, out_f, ensure_ascii=False, indent=2)

        num_exported += 1

    print(f"[INFO] Exported {num_exported} items for dataset {dataset} into {out_dir}")


def main() -> None:
    for dataset in DATASETS:
        export_dataset(dataset)


if __name__ == "__main__":
    main()
