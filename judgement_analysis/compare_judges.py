import argparse
import json
import os
from typing import Any, Dict, Iterable, Tuple


def iter_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                yield obj


def key_for(obj: Dict[str, Any]) -> Tuple[Any, Any]:
    """Build a stable key to align samples across judge runs."""
    # Prefer explicit ids if available, otherwise fall back to question text.
    qid = obj.get("id") or obj.get("question_id")
    qtext = obj.get("Question") or obj.get("input") or obj.get("text")
    return (qid, qtext)


def load_scores(path: str) -> Dict[Tuple[Any, Any], float]:
    scores: Dict[Tuple[Any, Any], float] = {}
    for obj in iter_jsonl(path):
        score = obj.get("score")
        if isinstance(score, (int, float)):
            scores[key_for(obj)] = float(score)
    return scores


def summarize(scores: Iterable[float]) -> Dict[str, float]:
    vals = list(scores)
    if not vals:
        return {"count": 0, "min": float("nan"), "max": float("nan"), "avg": float("nan")}
    return {
        "count": len(vals),
        "min": min(vals),
        "max": max(vals),
        "avg": sum(vals) / len(vals),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare score distributions between two judge runs on the same "
            "dataset/model JSONL file."
        )
    )
    parser.add_argument(
        "--judgements-root",
        default="data/judgements",
        help="Root directory containing judge runs (default: data/judgements)",
    )
    parser.add_argument(
        "--judge-a",
        required=True,
        help="First judge run directory name (e.g. judge_gpt-5.1-2025-11-13)",
    )
    parser.add_argument(
        "--judge-b",
        required=True,
        help="Second judge run directory name (e.g. judge_gpt-5.1-2025-11-13-oldjudgeprompt)",
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="Dataset directory name (e.g. lightblue__tengu_bench)",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Model judgements filename (without or with .json)",
    )

    args = parser.parse_args()

    model_file = args.model
    if not model_file.endswith(".json"):
        model_file = model_file + ".json"

    path_a = os.path.join(args.judgements_root, args.judge_a, args.dataset, model_file)
    path_b = os.path.join(args.judgements_root, args.judge_b, args.dataset, model_file)

    if not os.path.exists(path_a):
        raise SystemExit(f"File not found for judge A: {path_a}")
    if not os.path.exists(path_b):
        raise SystemExit(f"File not found for judge B: {path_b}")

    scores_a = load_scores(path_a)
    scores_b = load_scores(path_b)

    keys = set(scores_a.keys()) & set(scores_b.keys())
    only_a = set(scores_a.keys()) - keys
    only_b = set(scores_b.keys()) - keys

    aligned_a = [scores_a[k] for k in keys]
    aligned_b = [scores_b[k] for k in keys]
    diffs = [scores_b[k] - scores_a[k] for k in keys]

    summary_a = summarize(aligned_a)
    summary_b = summarize(aligned_b)
    summary_diff = summarize(diffs)

    print(f"# Comparing judges for {args.dataset}/{model_file}")
    print(f"# Judge A: {args.judge_a}")
    print(f"# Judge B: {args.judge_b}")
    print()
    print("Aligned samples:", summary_a["count"])
    print("Only in A:", len(only_a))
    print("Only in B:", len(only_b))
    print()
    print("Judge A stats: "
          f"min={summary_a['min']:.3g} max={summary_a['max']:.3g} avg={summary_a['avg']:.3g}")
    print("Judge B stats: "
          f"min={summary_b['min']:.3g} max={summary_b['max']:.3g} avg={summary_b['avg']:.3g}")
    print("B - A diff stats: "
          f"min={summary_diff['min']:.3g} max={summary_diff['max']:.3g} avg={summary_diff['avg']:.3g}")


if __name__ == "__main__":
    main()

