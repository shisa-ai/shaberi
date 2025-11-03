#!/usr/bin/env python3
"""Interactive Textual TUI for browsing shaberi benchmark answers and judgements.

USAGE:
    pip install textual  # Install dependency first
    python shaberi-viewer.py [--judge JUDGE] [--model MODEL]

FEATURES:
- Browse model answers and LLM judge scores across 4 evaluation datasets
- Filter by evaluation dataset using horizontal tab buttons
- View questions, model answers, ground truth, evaluation criteria, and scores
- Filter to show only scored questions with 'f' key
- Vim-style search to filter models (press '/', type query like '114', ESC to clear)
- Default judge: gpt-4.1-2025-04-14

KEYBOARD SHORTCUTS:
    q - Quit
    f - Toggle filter to show only scored questions
    / - Search/filter models (vim-style)
    ESC - Clear search filter

TODO:
- Multi-judge comparison support - currently displays single judge (default: gpt-4.1-2025-04-14)
  Future enhancement: Add ability to compare scores from multiple judges side-by-side
- Interactive collapsible sections for ground truth and eval criteria
  Currently all content is shown; future enhancement: click to expand/collapse
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Optional, Sequence

from rich.console import Group
from rich.table import Table
from rich.text import Text

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
MODEL_ANSWERS_DIR = DATA_DIR / "model_answers"
JUDGEMENTS_DIR = DATA_DIR / "judgements"

# Default judge model
DEFAULT_JUDGE = "gpt-4.1-2025-04-14"

# Eval datasets configuration
EVAL_DATASETS = {
    "elyza__ELYZA-tasks-100": {"name": "ELYZA-100", "max_score": 5},
    "lightblue__tengu_bench": {"name": "Tengu", "max_score": 10},
    "shisa-ai__ja-mt-bench-1shot": {"name": "JA-MT", "max_score": 10},
    "yuzuai__rakuda-questions": {"name": "Rakuda", "max_score": 10},
}

try:  # Defer Textual import so users get a clear message if it's missing.
    from textual.app import App, ComposeResult
    from textual.containers import Container, Horizontal, Vertical, VerticalScroll
    from textual.reactive import reactive
    from textual.widgets import Button, Footer, Header, Input, ListItem, ListView, Select, Static
except ImportError as exc:  # pragma: no cover - executed only when Textual is absent
    App = None  # type: ignore[assignment]
    ComposeResult = None  # type: ignore[assignment]
    Container = Horizontal = Vertical = VerticalScroll = None  # type: ignore[assignment]
    reactive = None  # type: ignore[assignment]
    Button = Footer = Header = Input = ListItem = ListView = Select = Static = None  # type: ignore[assignment]
    TEXTUAL_IMPORT_ERROR = exc
else:
    TEXTUAL_IMPORT_ERROR = None


def display_model_name(safe_name: str) -> str:
    """Convert safe filename to display name (replace __ with /)"""
    return safe_name.replace("__", "/")


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Load JSONL file and return list of records"""
    if not path.exists():
        return []

    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def safe_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


@dataclass
class QuestionRecord:
    index: int
    question: str
    model_answer: str
    ground_truth: Optional[str]
    eval_aspect: Optional[str]
    score: Optional[int]
    raw: Dict[str, Any]

    @property
    def has_score(self) -> bool:
        return self.score is not None

    def get_score_status(self, max_score: int) -> str:
        """Return visual indicator for score"""
        if self.score is None:
            return "❔"

        ratio = self.score / max_score
        if ratio >= 0.8:
            return "✅"
        elif ratio >= 0.6:
            return "⚠️"
        else:
            return "❌"


@dataclass
class EvalRun:
    dataset: str
    dataset_display: str
    model: str
    judge: str
    questions: List[QuestionRecord]
    max_score: int

    @property
    def avg_score(self) -> Optional[float]:
        scores = [q.score for q in self.questions if q.score is not None]
        return mean(scores) if scores else None

    @property
    def num_scored(self) -> int:
        return sum(1 for q in self.questions if q.score is not None)

    @property
    def total_questions(self) -> int:
        return len(self.questions)


@dataclass
class ModelData:
    safe_name: str
    display_name: str
    judge: str
    eval_runs: Dict[str, EvalRun]  # keyed by dataset safe name

    @property
    def overall_avg(self) -> Optional[float]:
        """Calculate overall average across all evals (normalized to 0-10 scale)"""
        normalized_scores = []
        for run in self.eval_runs.values():
            if run.avg_score is not None:
                # Normalize to 10-point scale
                normalized = (run.avg_score / run.max_score) * 10
                normalized_scores.append(normalized)
        return mean(normalized_scores) if normalized_scores else None


def list_judges(judgements_dir: Path) -> List[str]:
    """List all available judge directories"""
    if not judgements_dir.exists():
        return [DEFAULT_JUDGE]

    judges = []
    for judge_dir in judgements_dir.iterdir():
        if judge_dir.is_dir() and judge_dir.name.startswith("judge_"):
            judge_name = judge_dir.name.replace("judge_", "")
            judges.append(judge_name)

    return sorted(judges) if judges else [DEFAULT_JUDGE]


def list_models(judge: str, judgements_dir: Path) -> List[str]:
    """List all models that have judgements from the specified judge"""
    judge_dir = judgements_dir / f"judge_{judge}"
    if not judge_dir.exists():
        return []

    models = set()
    for dataset_dir in judge_dir.iterdir():
        if not dataset_dir.is_dir():
            continue
        for judgement_file in dataset_dir.glob("*.json"):
            model_name = judgement_file.stem
            models.add(model_name)

    return sorted(models)


def load_model_data(
    model_safe_name: str,
    judge: str,
    judgements_dir: Path,
    answers_dir: Path,
) -> Optional[ModelData]:
    """Load all eval runs for a model with judgements from specified judge"""
    eval_runs = {}

    for dataset_safe, dataset_info in EVAL_DATASETS.items():
        # Try to load judgement file
        judgement_path = judgements_dir / f"judge_{judge}" / dataset_safe / f"{model_safe_name}.json"

        if not judgement_path.exists():
            # No judgement for this dataset, skip
            continue

        # Load judgements
        judgements = load_jsonl(judgement_path)
        if not judgements:
            continue

        # Load corresponding model answers (for reference, though judgements contain everything)
        questions = []
        for idx, record in enumerate(judgements):
            question_record = QuestionRecord(
                index=idx,
                question=record.get("Question", ""),
                model_answer=record.get("ModelAnswer", ""),
                ground_truth=record.get("output"),
                eval_aspect=record.get("eval_aspect"),
                score=safe_int(record.get("score")),
                raw=record,
            )
            questions.append(question_record)

        if questions:
            eval_run = EvalRun(
                dataset=dataset_safe,
                dataset_display=dataset_info["name"],
                model=model_safe_name,
                judge=judge,
                questions=questions,
                max_score=dataset_info["max_score"],
            )
            eval_runs[dataset_safe] = eval_run

    if not eval_runs:
        return None

    return ModelData(
        safe_name=model_safe_name,
        display_name=display_model_name(model_safe_name),
        judge=judge,
        eval_runs=eval_runs,
    )


def format_score(score: Optional[int], max_score: int) -> Text:
    """Format score with color coding"""
    if score is None:
        return Text("N/A", style="dim")

    ratio = score / max_score
    if ratio >= 0.8:
        style = "bold green"
    elif ratio >= 0.6:
        style = "yellow"
    else:
        style = "bold red"

    return Text(f"{score}/{max_score}", style=style)


def build_question_renderable(record: QuestionRecord, max_score: int) -> Group:
    """Build renderable for a single question"""
    # Icon and header
    icon = record.get_score_status(max_score)
    icon_style = "green" if icon == "✅" else "bold red" if icon == "❌" else "yellow"

    header = Text()
    header.append(icon + " ", style=icon_style)
    header.append(f"Question {record.index + 1}", style="bold cyan")
    if record.score is not None:
        header.append(" • ", style="dim")
        score_text = format_score(record.score, max_score)
        header.append(score_text)

    # Main content table
    table = Table.grid(padding=(0, 1))
    table.add_column(justify="right", width=14, style="bold", no_wrap=True)
    table.add_column(justify="center", width=1, style="dim", no_wrap=True)
    table.add_column(ratio=1)

    # Question
    table.add_row("Question", "|", Text(record.question, style="bold"))

    # Model Answer
    table.add_row("Model Answer", "|", Text(record.model_answer, style="cyan"))

    # Ground Truth (show first line, full content in dim style for space efficiency)
    if record.ground_truth:
        # Show abbreviated version if too long
        if len(record.ground_truth) > 200:
            lines = record.ground_truth.split('\n')
            preview = lines[0][:100] + "..." if len(lines[0]) > 100 else lines[0]
            table.add_row("Ground Truth", "|", Text(preview, style="dim green"))
        else:
            table.add_row("Ground Truth", "|", Text(record.ground_truth, style="dim green"))

    # Eval Criteria (show abbreviated version if too long)
    if record.eval_aspect:
        if len(record.eval_aspect) > 150:
            lines = record.eval_aspect.split('\n')
            preview = lines[0][:100] + "..." if len(lines[0]) > 100 else lines[0]
            table.add_row("Criteria", "|", Text(preview, style="dim magenta"))
        else:
            table.add_row("Criteria", "|", Text(record.eval_aspect, style="dim magenta"))

    return Group(header, table)


def build_eval_renderable(run: EvalRun) -> Group:
    """Build renderable for an entire eval run"""
    if not run.questions:
        return Group(Text("No questions found for this evaluation.", style="dim"))

    renderables: List[Any] = []
    for record in run.questions:
        renderables.append(build_question_renderable(record, run.max_score))
        renderables.append(Text(""))  # Spacing

    return Group(*renderables)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Interactive viewer for shaberi benchmark answers and judgements.")
    parser.add_argument("--judge", default=DEFAULT_JUDGE, help=f"Judge model name (default: {DEFAULT_JUDGE})")
    parser.add_argument("--model", help="Model name to preselect (safe or display name)")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR, help="Path to data directory")
    args = parser.parse_args(argv)

    judge = args.judge
    data_dir = args.data_dir
    judgements_dir = data_dir / "judgements"
    answers_dir = data_dir / "model_answers"

    # Check for textual
    if App is None or TEXTUAL_IMPORT_ERROR is not None:
        print("This viewer requires the 'textual' package. Install it with `pip install textual`.")
        if TEXTUAL_IMPORT_ERROR is not None:
            print(f"Import error: {TEXTUAL_IMPORT_ERROR}")
        return 1

    # Check if judge exists
    available_judges = list_judges(judgements_dir)
    if judge not in available_judges:
        print(f"Judge '{judge}' not found. Available judges: {', '.join(available_judges)}")
        return 1

    # Load models
    model_names = list_models(judge, judgements_dir)
    if not model_names:
        print(f"No models found with judgements from judge '{judge}'")
        return 1

    app = ShaberiViewerApp(
        judge=judge,
        model_names=model_names,
        judgements_dir=judgements_dir,
        answers_dir=answers_dir,
        preselect_model=args.model,
    )
    app.run()
    return 0


if App is not None:

    class EvalButton(Button):
        """Button for selecting an eval dataset"""
        def __init__(self, dataset_safe: str, dataset_display: str):
            super().__init__(dataset_display, id=f"eval-{dataset_safe}")
            self.dataset_safe = dataset_safe
            self.dataset_display = dataset_display

    class ModelListItem(ListItem):
        def __init__(self, model: ModelData):
            # Format model name with overall score if available
            label = model.display_name
            if model.overall_avg is not None:
                label += f" (avg: {model.overall_avg:.2f}/10)"
            super().__init__(Static(label))
            self.model = model

    class ShaberiViewerApp(App):
        CSS = """
        Screen {
            layout: vertical;
        }
        #body {
            layout: horizontal;
            height: 1fr;
        }
        #sidebar {
            width: 40;
            min-width: 32;
            height: 1fr;
            border: solid $surface-darken-1;
            padding: 1 0;
        }
        #judge-info {
            margin: 0 1 1 1;
            color: $text-muted;
        }
        #model-search {
            margin: 0 1 1 1;
            height: auto;
        }
        #model-list {
            height: 1fr;
            margin: 0 1 0 1;
            overflow: auto;
        }
        #main {
            layout: vertical;
            width: 1fr;
            height: 1fr;
            padding: 1;
        }
        #eval-selector {
            height: auto;
            margin-bottom: 1;
        }
        #eval-buttons {
            height: auto;
            padding: 0 0 1 0;
        }
        .eval-button {
            margin-right: 1;
        }
        #content-summary {
            min-height: 1;
            margin-bottom: 1;
        }
        #details-panel {
            height: 1fr;
            border: solid $surface-darken-1;
            padding: 0 1;
        }
        """

        BINDINGS = [
            ("q", "quit", "Quit"),
            ("f", "toggle_scored_only", "Toggle scored only"),
            ("/", "focus_search", "Search"),
            ("escape", "clear_search", "Clear search"),
        ]

        show_scored_only = reactive(False)
        search_query = reactive("")

        def __init__(
            self,
            judge: str,
            model_names: List[str],
            judgements_dir: Path,
            answers_dir: Path,
            preselect_model: Optional[str] = None,
        ) -> None:
            super().__init__()
            self.judge = judge
            self.model_names = model_names
            self.judgements_dir = judgements_dir
            self.answers_dir = answers_dir
            self.preselect_model = preselect_model
            self.models: List[ModelData] = []
            self.selected_model: Optional[ModelData] = None
            self.selected_eval: Optional[str] = None

        def compose(self) -> ComposeResult:
            yield Header(show_clock=True)
            with Horizontal(id="body"):
                with Vertical(id="sidebar"):
                    yield Static(f"Judge: {self.judge}", id="judge-info")
                    yield Input(placeholder="Search models (/)", id="model-search")
                    yield ListView(id="model-list")
                with Vertical(id="main"):
                    with Container(id="eval-selector"):
                        with Horizontal(id="eval-buttons"):
                            for dataset_safe, dataset_info in EVAL_DATASETS.items():
                                yield EvalButton(dataset_safe, dataset_info["name"])
                    yield Static("", id="content-summary")
                    with VerticalScroll(id="details-panel"):
                        yield Static("Select a model to view benchmarks.", id="details-content")
            yield Footer()

        def on_mount(self) -> None:
            self.title = f"Shaberi Benchmarks • Judge: {self.judge}"
            self._load_models()
            self._populate_models()

        def action_toggle_scored_only(self) -> None:
            self.show_scored_only = not self.show_scored_only

        def action_focus_search(self) -> None:
            """Focus the search input"""
            search_input = self.query_one("#model-search", Input)
            search_input.focus()

        def action_clear_search(self) -> None:
            """Clear the search query"""
            search_input = self.query_one("#model-search", Input)
            search_input.value = ""
            # Focus back to model list
            model_list = self.query_one("#model-list", ListView)
            model_list.focus()

        def watch_show_scored_only(self, _: bool) -> None:
            self._refresh_content()

        def watch_search_query(self, query: str) -> None:
            """Update model list when search query changes"""
            self._populate_models()

        def on_button_pressed(self, event: Button.Pressed) -> None:
            """Handle eval button clicks"""
            if isinstance(event.button, EvalButton):
                self._select_eval(event.button.dataset_safe)

        def on_input_changed(self, event: Input.Changed) -> None:
            """Handle search input changes"""
            if event.input.id == "model-search":
                self.search_query = event.value

        def on_list_view_highlighted(self, event: ListView.Highlighted) -> None:
            if isinstance(event.item, ModelListItem):
                self._select_model(event.item.model)

        def _load_models(self) -> None:
            """Load all model data"""
            self.models = []
            for model_name in self.model_names:
                model_data = load_model_data(
                    model_name,
                    self.judge,
                    self.judgements_dir,
                    self.answers_dir,
                )
                if model_data:
                    self.models.append(model_data)

            # Sort by overall average (descending)
            self.models.sort(key=lambda m: m.overall_avg or 0, reverse=True)

        def _populate_models(self) -> None:
            """Populate model list with optional search filtering"""
            model_list = self.query_one("#model-list", ListView)
            model_list.clear()

            if not self.models:
                model_list.append(ListItem(Static("No models found.")))
                return

            # Filter models based on search query
            filtered_models = self.models
            if self.search_query:
                query_lower = self.search_query.lower()
                filtered_models = [
                    m for m in self.models
                    if query_lower in m.safe_name.lower() or query_lower in m.display_name.lower()
                ]

            if not filtered_models:
                model_list.append(ListItem(Static(f"No models matching '{self.search_query}'")))
                return

            for model in filtered_models:
                model_list.append(ModelListItem(model))

            # Preselect model if specified (only on first load)
            if self.preselect_model and not self.search_query:
                index = self._find_model_index(self.preselect_model, filtered_models)
            else:
                index = 0

            index = max(0, min(index, len(filtered_models) - 1))
            model_list.index = index
            self._select_model(filtered_models[index])

        def _find_model_index(self, target: Optional[str], models: Optional[List[ModelData]] = None) -> int:
            if not target:
                return 0
            if models is None:
                models = self.models
            target_lower = target.lower()
            for idx, model in enumerate(models):
                if model.safe_name.lower() == target_lower or model.display_name.lower() == target_lower:
                    return idx
            return 0

        def _select_model(self, model: Optional[ModelData]) -> None:
            """Select a model and show its evals"""
            if model is None:
                self.selected_model = None
                self.selected_eval = None
                self._refresh_content()
                return

            self.selected_model = model

            # Auto-select first available eval
            if model.eval_runs:
                first_eval = list(model.eval_runs.keys())[0]
                self._select_eval(first_eval)
            else:
                self.selected_eval = None
                self._refresh_content()

        def _select_eval(self, eval_dataset: str) -> None:
            """Select an eval dataset to display"""
            if not self.selected_model:
                return

            if eval_dataset not in self.selected_model.eval_runs:
                return

            self.selected_eval = eval_dataset
            self._refresh_content()

            # Update button states
            for button in self.query("EvalButton"):
                if button.dataset_safe == eval_dataset:
                    button.variant = "primary"
                else:
                    button.variant = "default"

        def _refresh_content(self) -> None:
            """Refresh the content panel"""
            summary = self.query_one("#content-summary", Static)
            details = self.query_one("#details-content", Static)

            if not self.selected_model or not self.selected_eval:
                summary.update("Select a model and evaluation.")
                details.update(Text("Select a model and evaluation to display questions."))
                return

            run = self.selected_model.eval_runs.get(self.selected_eval)
            if not run:
                summary.update("No data for this evaluation.")
                details.update(Text("No data found for this evaluation."))
                return

            questions = run.questions
            if self.show_scored_only:
                questions = [q for q in questions if q.has_score]

            # Update summary
            summary_text = f"{run.dataset_display}: {len(questions)} questions"
            if self.show_scored_only:
                summary_text += " (scored only)"
            if run.avg_score is not None:
                summary_text += f" • Average: {run.avg_score:.2f}/{run.max_score}"
            summary.update(summary_text)

            # Update details
            if not questions:
                details.update(Text("No questions match the current filter."))
                return

            # Create a temporary EvalRun with filtered questions
            filtered_run = EvalRun(
                dataset=run.dataset,
                dataset_display=run.dataset_display,
                model=run.model,
                judge=run.judge,
                questions=questions,
                max_score=run.max_score,
            )

            details.update(build_eval_renderable(filtered_run))


if __name__ == "__main__":
    sys.exit(main())
