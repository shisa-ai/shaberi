# Shaberi Rewrite Plan

## Overview

Clean, maintainable implementation of Shaberi (Japanese LLM evaluation suite) that mirrors multieval's architecture and supports advanced evaluation workflows.

## Goals

- **Maintainability**: Clear separation of concerns, standardized interfaces
- **Extensibility**: Easy to add new benchmarks, passes, and evaluation modes
- **Reproducibility**: Complete manifest/metadata tracking for all runs
- **Statistical rigor**: Support multiple runs with aggregated statistics
- **Cost tracking**: Token count tracking for all LLM operations

## Required Functionality

### Core Evaluation Flow

- [x] **Answer Generation**: Generate model responses to benchmark questions
  - Support all 4 sub-benchmarks: ELYZA-tasks-100, Tengu-Bench, MT-Bench, Rakuda
  - Configurable parameters: temperature, max_tokens, frequency_penalty
  - Checkpoint/resume support for long-running generation
  - Error tracking and retry logic

- [x] **Judging**: LLM-as-judge scoring of model answers
  - Support multiple judge models (GPT-4.1, GPT-5.1, Gemini, etc.)
  - Capture full judge reasoning + score
  - Retry/fallback for failed judgements
  - Multi-judge support for robustness

- [x] **Aggregation**: Combine scores across benchmarks
  - Per-benchmark scores (ELYZA, Tengu, MT-Bench, Rakuda)
  - Overall average score
  - Multi-judge aggregation (when multiple judges used)

- [ ] **Additional Passes**: Post-processing modifiers
  - Cross-lingual token leakage detection (language mixing penalty)
  - Heuristic-based modifiers (repetition detection, output format validation)
  - Composable pipeline of passes

### Data Management

- [ ] **Standardized Results Structure**
  ```
  results/
    {model_name}/
      {run_tag}/
        manifest.json           # Run metadata
        artifacts/
          shaberi/
            answers/              # Raw model answers
              elyza.jsonl
              tengu.jsonl
              mt_bench.jsonl
              rakuda.jsonl
            judgements/           # Judge outputs
              judge_{model}/
                elyza.jsonl       # score + judge_output + tokens
                tengu.jsonl
                mt_bench.jsonl
                rakuda.jsonl
            passes/               # Additional pass outputs
              lang_mixing/
                elyza.jsonl       # Modified scores + reasoning
            normalized/           # Unified format
              answers.jsonl
              judgements.jsonl
              scores.jsonl
            metrics.json          # Final aggregated metrics
  ```

- [ ] **Manifest Tracking**
  - Model info: name, parameters, base_url, api_key hash
  - Generation config: temperature, max_tokens, frequency_penalty
  - Judge info: model(s) used, base_url, timestamps
  - Token counts: per-benchmark, per-stage (generation/judging/passes)
  - Run metadata: run_id, tag, timestamp, status
  - Benchmark versions/commits for reproducibility

- [ ] **Multi-run Support**
  - `--runs N` flag to generate N independent runs
  - Statistical aggregation: mean, std dev, margin of error
  - Per-run manifests + aggregated summary manifest
  - Tag format: `{base_tag}-run{N}` (e.g., `default-run1`, `default-run2`)

### Token Tracking

- [ ] **Token Count Collection**
  - Answer generation: prompt tokens + completion tokens per question
  - Judging: prompt tokens + completion tokens per judgement
  - Additional passes: tokens consumed by each pass
  - Aggregate by: benchmark, stage, total

- [ ] **Cost Calculation** (future)
  - Per-model pricing tables
  - Breakdown by stage: generation / judging / passes
  - Total estimated cost per run

## Architecture Design

### 1. Prompt Management (Jinja2 Templates)

**Directory structure:**
```
evals/shaberi/
  prompts/
    generation/
      elyza.jinja
      tengu.jinja
      mt_bench.jinja
      rakuda.jinja
    judging/
      elyza/
        primary.jinja         # Main judge prompt
        fallback.jinja        # Score extraction retry
        example.jinja         # Few-shot example
      tengu/
        primary.jinja
        fallback.jinja
        example.jinja
      mt_bench/
        primary.jinja
        fallback.jinja
      rakuda/
        primary.jinja
        fallback.jinja
    passes/
      lang_mixing/
        detection.jinja
        penalty.jinja
```

**Template interface:**
```python
from jinja2 import Environment, FileSystemLoader

class PromptManager:
    def __init__(self, prompts_dir: Path):
        self.env = Environment(loader=FileSystemLoader(prompts_dir))

    def render_generation_prompt(
        self,
        benchmark: str,
        question: str,
        **kwargs
    ) -> str:
        """Render question prompt for model answer generation."""
        template = self.env.get_template(f"generation/{benchmark}.jinja")
        return template.render(question=question, **kwargs)

    def render_judge_prompt(
        self,
        benchmark: str,
        question: str,
        answer: str,
        criteria: str | None = None,
        examples: list | None = None,
        **kwargs
    ) -> str:
        """Render judge evaluation prompt."""
        template = self.env.get_template(f"judging/{benchmark}/primary.jinja")
        return template.render(
            question=question,
            answer=answer,
            criteria=criteria,
            examples=examples,
            **kwargs
        )

    def render_fallback_prompt(
        self,
        benchmark: str,
        question: str,
        answer: str,
        judge_output: str,
        scale: str,
        **kwargs
    ) -> str:
        """Render fallback score extraction prompt."""
        template = self.env.get_template(f"judging/{benchmark}/fallback.jinja")
        return template.render(
            question=question,
            answer=answer,
            judge_output=judge_output,
            scale=scale,
            **kwargs
        )
```

**Benefits:**
- Separation of prompt logic from code
- Easy to version control and review prompt changes
- Support for includes/macros for common instructions
- Template inheritance for shared structure

### 2. Execution Pipeline

**Clean executables mirroring multieval:**

```bash
# Generate answers + judge + aggregate
./run-evals --model shisa-ai/shisa-v1 --benchmark all --tag default --runs 3

# Rejudge with different judge model
./run-evals --model shisa-ai/shisa-v1 --benchmark all --tag default --rejudge --judge-model gpt-5.1

# Apply additional passes (modifiers)
./run-evals --model shisa-ai/shisa-v1 --benchmark all --tag default --passes lang_mixing

# View aggregated scores
./view-scores --tag default

# Interactive inspection
./inspect-output --tag default
```

**Modular stages:**
```python
class ShaberiPipeline:
    def __init__(self, config: ShaberiConfig):
        self.config = config
        self.prompt_manager = PromptManager(config.prompts_dir)
        self.result_manager = ResultManager(config.results_dir)

    def run(self, model: str, benchmarks: list[str], tag: str, runs: int = 1):
        """Main execution pipeline."""
        for run_num in range(1, runs + 1):
            run_tag = f"{tag}-run{run_num}" if runs > 1 else tag

            # Stage 1: Generate answers
            if not self.config.skip_generate:
                self.generate_answers(model, benchmarks, run_tag)

            # Stage 2: Judge answers
            if not self.config.skip_judge:
                self.judge_answers(model, benchmarks, run_tag)

            # Stage 3: Apply additional passes
            if self.config.passes:
                self.apply_passes(model, benchmarks, run_tag, self.config.passes)

            # Stage 4: Aggregate and normalize
            self.aggregate_results(model, benchmarks, run_tag)

        # If multiple runs, compute statistics
        if runs > 1:
            self.aggregate_multi_run_stats(model, tag, runs)

    def generate_answers(self, model: str, benchmarks: list[str], tag: str):
        """Generate model answers with token tracking."""
        manifest = self.result_manager.init_manifest(model, tag)

        for benchmark in benchmarks:
            dataset = self.load_benchmark_dataset(benchmark)
            generator = AnswerGenerator(
                model=model,
                prompt_manager=self.prompt_manager,
                benchmark=benchmark,
                config=self.config.generation_config,
            )

            answers = []
            total_tokens = {"prompt": 0, "completion": 0}

            for question in dataset:
                result = generator.generate(question)
                answers.append(result.answer)
                total_tokens["prompt"] += result.prompt_tokens
                total_tokens["completion"] += result.completion_tokens

            # Save answers
            self.result_manager.save_answers(model, tag, benchmark, answers)

            # Update manifest
            manifest.update_generation_stats(
                benchmark=benchmark,
                tokens=total_tokens,
                timestamp=datetime.utcnow(),
            )

        manifest.save()

    def judge_answers(self, model: str, benchmarks: list[str], tag: str):
        """Judge answers with full reasoning capture + token tracking."""
        manifest = self.result_manager.load_manifest(model, tag)
        judge_model = self.config.judge_model

        for benchmark in benchmarks:
            answers = self.result_manager.load_answers(model, tag, benchmark)
            judge = Judge(
                judge_model=judge_model,
                prompt_manager=self.prompt_manager,
                benchmark=benchmark,
                config=self.config.judge_config,
            )

            judgements = []
            total_tokens = {"prompt": 0, "completion": 0}

            for answer in answers:
                result = judge.evaluate(answer)
                judgements.append({
                    "id": answer.id,
                    "score": result.score,
                    "judge_output": result.reasoning,
                    "prompt_tokens": result.prompt_tokens,
                    "completion_tokens": result.completion_tokens,
                })
                total_tokens["prompt"] += result.prompt_tokens
                total_tokens["completion"] += result.completion_tokens

            # Save judgements
            self.result_manager.save_judgements(
                model, tag, benchmark, judge_model, judgements
            )

            # Update manifest
            manifest.update_judge_stats(
                benchmark=benchmark,
                judge_model=judge_model,
                tokens=total_tokens,
                timestamp=datetime.utcnow(),
            )

        manifest.save()

    def apply_passes(
        self,
        model: str,
        benchmarks: list[str],
        tag: str,
        passes: list[str]
    ):
        """Apply additional evaluation passes (modifiers)."""
        manifest = self.result_manager.load_manifest(model, tag)

        for pass_name in passes:
            pass_module = self.load_pass(pass_name)

            for benchmark in benchmarks:
                answers = self.result_manager.load_answers(model, tag, benchmark)
                judgements = self.result_manager.load_judgements(
                    model, tag, benchmark, self.config.judge_model
                )

                # Apply pass (e.g., language mixing detection)
                result = pass_module.apply(
                    answers=answers,
                    judgements=judgements,
                    benchmark=benchmark,
                    config=self.config,
                )

                # Save pass output
                self.result_manager.save_pass_output(
                    model, tag, benchmark, pass_name, result.modified_judgements
                )

                # Update manifest with pass metadata
                manifest.update_pass_stats(
                    benchmark=benchmark,
                    pass_name=pass_name,
                    tokens=result.tokens,
                    modifications=result.stats,
                    timestamp=datetime.utcnow(),
                )

        manifest.save()
```

### 3. Additional Passes (Modifiers)

**Pass interface:**
```python
from dataclasses import dataclass
from typing import Protocol

@dataclass
class PassResult:
    """Result of applying an evaluation pass."""
    modified_judgements: list[dict]  # Updated judgements
    stats: dict  # Pass-specific statistics
    tokens: dict  # Token usage if LLM-based

class EvaluationPass(Protocol):
    """Interface for evaluation passes."""

    def apply(
        self,
        answers: list[Answer],
        judgements: list[Judgement],
        benchmark: str,
        config: dict,
    ) -> PassResult:
        """Apply pass to modify judgements."""
        ...

class LanguageMixingPass:
    """Detect and penalize cross-lingual token leakage."""

    def __init__(self, prompt_manager: PromptManager):
        self.prompt_manager = prompt_manager

    def apply(
        self,
        answers: list[Answer],
        judgements: list[Judgement],
        benchmark: str,
        config: dict,
    ) -> PassResult:
        modified = []
        stats = {"total": len(answers), "penalized": 0, "avg_penalty": 0.0}
        total_tokens = {"prompt": 0, "completion": 0}

        for answer, judgement in zip(answers, judgements):
            # Detect language mixing
            detection_result = self.detect_mixing(answer.output, config)

            if detection_result.has_mixing:
                # Apply penalty
                penalty = self.calculate_penalty(
                    mixing_ratio=detection_result.mixing_ratio,
                    config=config,
                )
                modified_score = max(0, judgement.score - penalty)

                # Add reasoning
                modified_reasoning = (
                    f"[Language Mixing Penalty: -{penalty:.2f}]\n"
                    f"{detection_result.explanation}\n\n"
                    f"Original judge output:\n{judgement.judge_output}"
                )

                modified.append({
                    **judgement.dict(),
                    "score": modified_score,
                    "judge_output": modified_reasoning,
                    "lang_mixing_penalty": penalty,
                    "lang_mixing_ratio": detection_result.mixing_ratio,
                })

                stats["penalized"] += 1
                stats["avg_penalty"] += penalty
            else:
                modified.append(judgement.dict())

            total_tokens["prompt"] += detection_result.prompt_tokens
            total_tokens["completion"] += detection_result.completion_tokens

        if stats["penalized"] > 0:
            stats["avg_penalty"] /= stats["penalized"]

        return PassResult(
            modified_judgements=modified,
            stats=stats,
            tokens=total_tokens,
        )

    def detect_mixing(self, text: str, config: dict):
        """Detect language mixing in output."""
        # Could be heuristic (regex/charset detection) or LLM-based
        # If LLM-based, use prompt template and track tokens
        pass
```

**Example passes:**
- `lang_mixing`: Cross-lingual token leakage detection
- `repetition`: Detect and penalize repetitive outputs
- `format_validation`: Check output format compliance
- `safety_check`: Flag potentially unsafe content

### 4. Token Tracking

**Token counter interface:**
```python
@dataclass
class TokenUsage:
    """Token usage for an LLM call."""
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

    @property
    def cost_estimate(self, model_pricing: dict) -> float:
        """Estimate cost based on model pricing."""
        input_cost = self.prompt_tokens * model_pricing["input_per_1k"] / 1000
        output_cost = self.completion_tokens * model_pricing["output_per_1k"] / 1000
        return input_cost + output_cost

class TokenTracker:
    """Track token usage across all stages."""

    def __init__(self):
        self.usage = {
            "generation": {},  # {benchmark: TokenUsage}
            "judging": {},     # {benchmark: {judge_model: TokenUsage}}
            "passes": {},      # {pass_name: {benchmark: TokenUsage}}
        }

    def add_generation(self, benchmark: str, tokens: TokenUsage):
        self.usage["generation"][benchmark] = tokens

    def add_judging(self, benchmark: str, judge_model: str, tokens: TokenUsage):
        if benchmark not in self.usage["judging"]:
            self.usage["judging"][benchmark] = {}
        self.usage["judging"][benchmark][judge_model] = tokens

    def add_pass(self, pass_name: str, benchmark: str, tokens: TokenUsage):
        if pass_name not in self.usage["passes"]:
            self.usage["passes"][pass_name] = {}
        self.usage["passes"][pass_name][benchmark] = tokens

    def total(self) -> TokenUsage:
        """Aggregate total token usage."""
        total_prompt = 0
        total_completion = 0

        for tokens in self.usage["generation"].values():
            total_prompt += tokens.prompt_tokens
            total_completion += tokens.completion_tokens

        for judge_data in self.usage["judging"].values():
            for tokens in judge_data.values():
                total_prompt += tokens.prompt_tokens
                total_completion += tokens.completion_tokens

        for pass_data in self.usage["passes"].values():
            for tokens in pass_data.values():
                total_prompt += tokens.prompt_tokens
                total_completion += tokens.completion_tokens

        return TokenUsage(
            prompt_tokens=total_prompt,
            completion_tokens=total_completion,
            total_tokens=total_prompt + total_completion,
        )

    def to_dict(self) -> dict:
        """Serialize for manifest."""
        return {
            "generation": {
                bench: {
                    "prompt_tokens": tokens.prompt_tokens,
                    "completion_tokens": tokens.completion_tokens,
                    "total_tokens": tokens.total_tokens,
                }
                for bench, tokens in self.usage["generation"].items()
            },
            "judging": {
                bench: {
                    judge: {
                        "prompt_tokens": tokens.prompt_tokens,
                        "completion_tokens": tokens.completion_tokens,
                        "total_tokens": tokens.total_tokens,
                    }
                    for judge, tokens in judge_data.items()
                }
                for bench, judge_data in self.usage["judging"].items()
            },
            "passes": {
                pass_name: {
                    bench: {
                        "prompt_tokens": tokens.prompt_tokens,
                        "completion_tokens": tokens.completion_tokens,
                        "total_tokens": tokens.total_tokens,
                    }
                    for bench, tokens in pass_data.items()
                }
                for pass_name, pass_data in self.usage["passes"].items()
            },
            "total": {
                "prompt_tokens": self.total().prompt_tokens,
                "completion_tokens": self.total().completion_tokens,
                "total_tokens": self.total().total_tokens,
            }
        }
```

### 5. Multi-Run Statistics

**Statistical aggregation:**
```python
from dataclasses import dataclass
import numpy as np
from scipy import stats as scipy_stats

@dataclass
class BenchmarkStats:
    """Statistics for a benchmark across multiple runs."""
    mean: float
    std_dev: float
    margin_of_error: float  # 95% confidence interval
    min: float
    max: float
    runs: int
    values: list[float]

class MultiRunAggregator:
    """Aggregate statistics across multiple runs."""

    def aggregate(
        self,
        model: str,
        base_tag: str,
        runs: int,
        benchmarks: list[str],
    ) -> dict:
        """Compute statistics across N runs."""
        results = {
            "model": model,
            "tag": base_tag,
            "runs": runs,
            "benchmarks": {},
            "overall": None,
        }

        # Collect scores for each benchmark
        benchmark_scores = {bench: [] for bench in benchmarks}

        for run_num in range(1, runs + 1):
            run_tag = f"{base_tag}-run{run_num}"
            manifest = self.result_manager.load_manifest(model, run_tag)

            for bench in benchmarks:
                score = manifest.metrics[bench]["score"]
                benchmark_scores[bench].append(score)

        # Compute stats for each benchmark
        for bench, scores in benchmark_scores.items():
            results["benchmarks"][bench] = self._compute_stats(scores)

        # Compute overall stats (average of benchmark means)
        overall_scores = []
        for run_num in range(1, runs + 1):
            run_tag = f"{base_tag}-run{run_num}"
            manifest = self.result_manager.load_manifest(model, run_tag)
            overall_scores.append(manifest.metrics["overall_score"])

        results["overall"] = self._compute_stats(overall_scores)

        return results

    def _compute_stats(self, values: list[float]) -> BenchmarkStats:
        """Compute statistics for a list of values."""
        arr = np.array(values)
        mean = np.mean(arr)
        std_dev = np.std(arr, ddof=1) if len(arr) > 1 else 0.0

        # 95% confidence interval using t-distribution
        if len(arr) > 1:
            sem = scipy_stats.sem(arr)
            margin = sem * scipy_stats.t.ppf(0.975, len(arr) - 1)
        else:
            margin = 0.0

        return BenchmarkStats(
            mean=float(mean),
            std_dev=float(std_dev),
            margin_of_error=float(margin),
            min=float(np.min(arr)),
            max=float(np.max(arr)),
            runs=len(arr),
            values=[float(v) for v in values],
        )
```

### 6. Viewer Integration

**Unified viewer command:**
```bash
# View aggregated scores with statistics
./view-scores --tag default

# Example output:
# Model: shisa-ai/shisa-v1
# Tag: default (3 runs)
#
# Benchmark          Mean    Std Dev   95% CI        Range
# ─────────────────────────────────────────────────────────
# ELYZA-tasks-100    7.23    0.15      ±0.31        7.05-7.38
# Tengu-Bench        6.89    0.22      ±0.45        6.64-7.09
# MT-Bench           7.45    0.11      ±0.23        7.35-7.57
# Rakuda             7.12    0.18      ±0.37        6.91-7.28
# ─────────────────────────────────────────────────────────
# Overall            7.17    0.12      ±0.25        7.04-7.28
#
# Token Usage:
# Generation:  1.2M tokens  ($1.20 estimated)
# Judging:     2.4M tokens  ($2.40 estimated)
# Passes:      0.3M tokens  ($0.30 estimated)
# Total:       3.9M tokens  ($3.90 estimated)
```

## Implementation Phases

### Phase 1: Core Refactor
- [ ] Migrate to Jinja2 templates for all prompts
- [ ] Implement clean pipeline with separated stages
- [ ] Standardize results/ directory structure
- [ ] Add comprehensive manifest tracking

### Phase 2: Token Tracking
- [ ] Implement TokenUsage tracking in all LLM calls
- [ ] Add token counts to manifests
- [ ] Build cost estimation framework

### Phase 3: Multi-Run Support
- [ ] Implement --runs flag
- [ ] Build statistical aggregation
- [ ] Update viewers for multi-run display

### Phase 4: Additional Passes
- [ ] Build pass framework
- [ ] Implement language mixing pass
- [ ] Add repetition detection pass

### Phase 5: Tooling
- [ ] Create run-evals executable
- [ ] Create view-scores executable
- [ ] Update inspect-output for Shaberi

## Migration Path

1. Keep existing Shaberi working alongside new implementation
2. Build new implementation in `evals/shaberi-v2/`
3. Test new implementation thoroughly
4. Migrate data format converters
5. Switch default to new implementation
6. Deprecate old implementation

## Open Questions

1. **Judge model fallback strategy**: How to handle multiple judge models in a single run?
2. **Pass composition**: Should passes be composable (output of one feeds into another)?
3. **Dataset versioning**: How to track benchmark dataset versions for reproducibility?
4. **Incremental re-evaluation**: Can we rejudge only specific benchmarks without full rerun?
5. **Distributed execution**: Support for parallel execution across multiple GPUs/machines?

## Additional Considerations

- **Align with multieval manifests and viewer**  
  - Reuse the existing `results/<model-safe>/<tag>/manifest.json` schema and `artifacts/shaberi/...` layout so Shaberi looks like any other eval to multieval and `inspect-output`.  
  - Ensure normalized `answers.jsonl` / `judgements.jsonl` / `scores.jsonl` for Shaberi match multieval’s unified record schemas (including `answer_id`, `judge_model`, `reasoning`, optional `judge_output`, and token fields).

- **Judge output and reasoning storage**  
  - Continue to record the full judge analysis string (`judge_output`) per sample, but consider whether the normalized `reasoning` field should store the full text, a trimmed version, or a summarized view to keep TUI panes readable.  
  - For very large runs, consider an optional “compact mode” that keeps only `reasoning` plus a pointer (path) to the raw judge transcript in artifacts.

- **Reprocess vs rejudge semantics**  
  - Clearly separate “reprocess” (parse existing Shaberi outputs into normalized artifacts/metrics) from “rejudge” (re-run LLM judges on existing answers), mirroring multieval’s `--reprocess` / `--rejudge` behavior.  
  - Document which fields can be regenerated without touching the endpoint (e.g., aggregation, passes, metrics) and which require hitting model/judge APIs again.

- **Passes and provenance**  
  - When passes modify scores (e.g., language-mixing penalties), always keep the original score and record pass-specific deltas and rationale as structured fields (e.g., `leakage_penalty`, `leakage_reason`) instead of overwriting scores in-place.  
  - Include pass configuration and version info in the manifest so multi-run stats can distinguish “baseline” vs “baseline + passes”.

- **Multi-run tagging strategy**  
  - Prefer tags-per-run (e.g., `default-run1`, `default-run2`, …) and keep a small Shaberi-specific aggregator that computes multi-run stats across tags, instead of overloading a single manifest with multiple internal runs.  
  - Make sure `view-scores` can operate both on a single tag and on a base tag + `--runs N` pattern, and that manifests record which tags were included in an aggregated report.

- **Token accounting consistency**  
  - Standardize token fields across stages: e.g., `answer_prompt_tokens`, `answer_completion_tokens`, `judge_prompt_tokens`, `judge_completion_tokens`, and analogous fields for passes.  
  - Ensure token counts are recorded at the per-sample level and rolled up into per-benchmark and per-run aggregates in the manifest so later cost analysis scripts don’t need to re-read all JSONL.

- **Prompt versioning and A/B testing**  
  - Treat the repo git commit recorded in the manifest as the primary “prompt+code version” for reproducibility, optionally tagging human-readable prompt branches (e.g., `shaberi-prompts-v3`) when you make significant prompt changes.  
  - If needed later, add a lightweight per-benchmark `prompt_id` or template hash in the manifest to make prompt drift visible at a glance and to support A/B judge prompt experiments with separate metrics and manifests.

- **Legacy data compatibility**  
  - Plan and document a one-time migration script that can take existing Shaberi `data/model_answers` and `data/judgements` trees and emit v2-style normalized artifacts under `results/`, so historical runs remain explorable in `inspect-output`.  
  - For missing fields in older runs (e.g., no `judge_output` / token counts), define explicit defaults and make the viewer robust to partial data.

- **Failure handling and partial runs**  
  - Mirror multieval’s behavior where manifests are updated incrementally and partially completed runs (failed benchmarks, judge timeouts) still produce a valid manifest with `status=error` and per-eval `status`/`error` fields.  
  - Make `view-scores` and `inspect-output` resilient to missing or errored benchmarks, and clearly surface which parts of Shaberi completed vs failed.

- **API client reuse and rate limiting**  
  - Consider sharing or at least modeling Shaberi’s OpenAI/Gemini clients after multieval’s client wrappers, including concurrency and retry/backoff behavior, so rate limiting and transient errors are handled uniformly.  
  - If Shaberi and multieval may run concurrently against the same judge endpoint, document or implement a simple coordination mechanism for max judge concurrency.  
