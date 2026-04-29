import argparse
import json
import os
from typing import Any, Dict, Iterable, List


def iter_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    """Yield JSON objects from a JSONL/ndjson file, skipping malformed lines."""
    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                print(f"# WARNING: JSON parse error in {path}:{lineno}", flush=True)
                continue
            if isinstance(obj, dict):
                yield obj


def shorten(text: Any, max_len: int = 160) -> str:
    """Collapse whitespace and truncate text for compact CLI viewing."""
    if text is None:
        return ""
    s = str(text).replace("\n", " ").replace("\r", " ")
    s = " ".join(s.split())
    return s if len(s) <= max_len else s[: max_len - 3] + "..."


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Scan a judgement JSONL file and print examples with low scores "
            "for manual inspection."
        )
    )
    parser.add_argument(
        "--judgements-root",
        default="data/judgements",
        help="Root directory containing judge runs (default: data/judgements)",
    )
    parser.add_argument(
        "--judge",
        required=True,
        help="Judge run directory name (e.g. judge_gpt-5.1-2025-11-13)",
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="Dataset directory name under the judge (e.g. lightblue__tengu_bench)",
    )
    parser.add_argument(
        "--model",
        required=True,
        help=(
            "Model judgements filename under the dataset "
            "(e.g. shisa-ai__183-llama3.1-8b-v2.1o-dpo-7e8.json)"
        ),
    )
    parser.add_argument(
        "--max-score",
        type=float,
        default=5.0,
        help="Show entries with score <= this value (default: 5)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Maximum number of examples to print (default: 20; 0 = no limit)",
    )
    parser.add_argument(
        "--show-answer",
        action="store_true",
        help="Print a truncated snippet of the model answer for context",
    )
    parser.add_argument(
        "--show-judge",
        action="store_true",
        help="Print a truncated snippet of the judge_output analysis",
    )

    args = parser.parse_args()

    model_file = args.model
    if not model_file.endswith(".json"):
        model_file = model_file + ".json"

    path = os.path.join(args.judgements_root, args.judge, args.dataset, model_file)
    if not os.path.exists(path):
        raise SystemExit(f"Judgement file not found: {path}")

    count_total = 0
    count_low = 0
    printed = 0

    print(
        f"# Scanning {path} for scores <= {args.max_score} "
        f"(limit={args.limit or 'none'})",
        flush=True,
    )

    for obj in iter_jsonl(path):
        count_total += 1
        score = obj.get("score")
        if not isinstance(score, (int, float)):
            continue
        if score > args.max_score:
            continue
        count_low += 1

        if args.limit and printed >= args.limit:
            continue

        ident_parts: List[str] = []
        if "id" in obj:
            ident_parts.append(f"id={obj['id']}")
        if "question_id" in obj:
            ident_parts.append(f"qid={obj['question_id']}")
        if "Category" in obj:
            ident_parts.append(f"cat={shorten(obj['Category'], 40)}")
        if "category" in obj and "Category" not in obj:
            ident_parts.append(f"cat={shorten(obj['category'], 40)}")

        ident = " ".join(ident_parts) if ident_parts else "(no id)"
        print(f"\n## score={score}  {ident}")

        if args.show_answer:
            answer = (
                obj.get("ModelAnswer")
                or obj.get("Answer")
                or obj.get("output")
                or obj.get("response")
                or obj.get("answer")
            )
            print(f"- answer: {shorten(answer)}")

        if args.show_judge:
            judge = obj.get("judge_output")
            print(f"- judge:  {shorten(judge)}")

        printed += 1

    print(
        f"\n# Done. total={count_total} low={count_low} "
        f"(printed={printed}, threshold={args.max_score})",
        flush=True,
    )


if __name__ == "__main__":
    main()

