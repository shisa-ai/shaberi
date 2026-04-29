#!/usr/bin/env python3
"""
Fill in English translations for exported eval items using a local (or remote)
OpenAI-compatible LLM, typically a strong JP→EN translation model.

Usage (typical):

  python judgement_analysis/translate_export_yaml_with_llm.py \\
      --base-url http://localhost:8000/v1 \\
      --dataset all

By default, the script:
  - Discovers the default model by calling GET /models on the given base URL.
  - Fills in, per question file:
      question.translation_en
      answer.gold_backtranslation_en
      rubric.explanation_en
  - Skips fields that already have a non-empty string (to avoid overwriting).

You can override the model and force retranslation:

  python judgement_analysis/translate_export_yaml_with_llm.py \\
      --base-url http://localhost:8000/v1 \\
      --model shisa-ai/your-translation-model \\
      --dataset lightblue__tengu_bench \\
      --retranslate

The script stores the translation model metadata in each file under:

  translation_meta: { model, base_url, last_updated }

so we can later tell which model produced the current translations.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path
from typing import List

from openai import OpenAI

try:
    import yaml  # type: ignore[import]
except Exception as e:  # pragma: no cover - environment-specific
    raise SystemExit(
        "PyYAML is required for translate_export_yaml_with_llm.py. "
        "Install it in the shaberi environment, e.g.: "
        "`mamba run -n shaberi mamba install pyyaml`."
    ) from e


ROOT = Path(__file__).resolve().parents[1]

DATASETS: List[str] = [
    "elyza__ELYZA-tasks-100",
    "lightblue__tengu_bench",
    "shisa-ai__ja-mt-bench-1shot",
    "yuzuai__rakuda-questions",
]


def discover_model(base_url: str, api_key: str) -> str:
    """
    Query the /models endpoint and pick a default model ID.

    This assumes an OpenAI-compatible server (e.g., vLLM) at base_url.
    """
    client = OpenAI(api_key=api_key, base_url=base_url)
    resp = client.models.list()
    if not getattr(resp, "data", None):
        raise RuntimeError(f"No models returned from {base_url}/models")
    model_id = resp.data[0].id
    print(f"[INFO] Using discovered translation model: {model_id}")
    return model_id


def translate_text(
    client: OpenAI,
    model: str,
    text: str,
    purpose: str,
) -> str:
    """
    Call the chat completions API to translate JP→EN for documentation.
    """
    if not text:
        return text

    system_prompt = (
        "You are a professional translator from Japanese to English. "
        "Translate the user-provided text into natural, clear English suitable "
        "for evaluation documentation. Preserve meaning and important details. "
        "Do not add commentary or explanations; output only the translation."
    )
    user_prompt = f"[Purpose: {purpose}]\n\n{text}"

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.0,
        max_tokens=2048,
    )
    return resp.choices[0].message.content.strip()


def process_file(
    path: Path,
    client: OpenAI,
    model: str,
    base_url_str: str,
    retranslate: bool,
) -> None:
    """
    Load a single qXXX.yaml (JSON) file, fill translations, and write back.
    """
    with path.open("r", encoding="utf-8") as f:
        # Files may currently be JSON or YAML; safe_load handles both.
        data = yaml.safe_load(f)

    changed = False

    # Question text
    q = data.get("question", {})
    orig_q = q.get("original")
    trans_q = q.get("translation_en")
    if orig_q and (retranslate or not isinstance(trans_q, str) or not trans_q.strip()):
        print(f"[INFO] Translating question for {path.name}")
        q["translation_en"] = translate_text(client, model, orig_q, "question")
        data["question"] = q
        changed = True

    # Gold answer back-translation
    ans = data.get("answer", {})
    gold = ans.get("gold_original")
    gold_bt = ans.get("gold_backtranslation_en")
    if gold and (retranslate or not isinstance(gold_bt, str) or not gold_bt.strip()):
        print(f"[INFO] Translating gold answer for {path.name}")
        ans["gold_backtranslation_en"] = translate_text(
            client, model, gold, "gold_answer"
        )
        data["answer"] = ans
        changed = True

    # Rubric explanation
    rub = data.get("rubric", {})
    rub_orig = rub.get("original")
    rub_exp = rub.get("explanation_en")
    if rub_orig and (retranslate or not isinstance(rub_exp, str) or not rub_exp.strip()):
        print(f"[INFO] Translating rubric for {path.name}")
        rub["explanation_en"] = translate_text(
            client, model, rub_orig, "rubric"
        )
        data["rubric"] = rub
        changed = True

    # Translation metadata
    if changed or retranslate:
        meta = data.get("translation_meta", {}) or {}
        meta["model"] = model
        meta["base_url"] = base_url_str
        meta["last_updated"] = dt.datetime.now(dt.timezone.utc).isoformat()
        data["translation_meta"] = meta
        changed = True

    if changed:
        with path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(
                data,
                f,
                allow_unicode=True,
                sort_keys=False,
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fill English translations in exported eval YAML files using a local OpenAI-compatible LLM.",
    )
    parser.add_argument(
        "--base-url",
        default="http://localhost:8000/v1",
        help="OpenAI-compatible base URL (default: http://localhost:8000/v1)",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="API key for the OpenAI-compatible server (default: OPENAI_API_KEY env or 'EMPTY').",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model name to use for translation. "
             "If omitted, the script queries /models at base-url and uses the first model.",
    )
    parser.add_argument(
        "--dataset",
        choices=DATASETS + ["all"],
        default="all",
        help="Which dataset to process (default: all).",
    )
    parser.add_argument(
        "--max-items",
        type=int,
        default=None,
        help="Maximum number of items to process per dataset (for testing).",
    )
    parser.add_argument(
        "--retranslate",
        action="store_true",
        help="If set, overwrite existing translations instead of only filling missing ones.",
    )

    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("OPENAI_API_KEY") or "EMPTY"

    # Discover model if not provided
    model = args.model
    if model is None:
        try:
            model = discover_model(args.base_url, api_key)
        except Exception as e:
            raise SystemExit(f"[ERROR] Could not discover model from {args.base_url}: {e}")

    client = OpenAI(api_key=api_key, base_url=args.base_url)

    datasets: List[str]
    if args.dataset == "all":
        datasets = DATASETS
    else:
        datasets = [args.dataset]

    for dataset in datasets:
        export_dir = ROOT / "judgement_analysis" / "export" / dataset
        if not export_dir.exists():
            print(f"[WARN] Export directory does not exist for dataset {dataset}: {export_dir}")
            continue

        files = sorted(export_dir.glob("q*.yaml"))
        if not files:
            print(f"[WARN] No q*.yaml files found for dataset {dataset} in {export_dir}")
            continue

        n_files = len(files)
        if args.max_items is not None:
            n_files = min(n_files, args.max_items)
        print(f"[INFO] Processing {n_files} items for dataset {dataset} using model {model}")
        for idx, path in enumerate(files, start=1):
            if args.max_items is not None and idx > args.max_items:
                break
            try:
                process_file(path, client, model, args.base_url, args.retranslate)
            except Exception as e:
                print(f"[WARN] Failed to translate {path}: {e}")


if __name__ == "__main__":
    main()
