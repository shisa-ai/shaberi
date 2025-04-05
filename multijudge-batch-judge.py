#!/usr/bin/env python3
"""
Script to rejudge models using three different LLM judges.
Skips models that have already been judged and shows progress in a TUI.
"""

import os
import sys
import time
import signal
import threading
import subprocess
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Set

import pandas as pd
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn
from rich.layout import Layout

# Constants
JUDGES = ['llmjudge-tulu405', 'llmjudge-llama33', 'llmjudge-athenev2']
TESTS = ['lightblue__tengu_bench', 'elyza__ELYZA-tasks-100', 'shisa-ai__ja-mt-bench-1shot', 'yuzuai__rakuda-questions']

class JudgeState:
    def __init__(self, name: str, models: List[str]):
        self.name = name
        self.total_models = len(models)
        self.remaining_models = models.copy()
        self.completed = 0
        self.current_model = ""
        self.last_run_time = ""
        self.status = "Waiting to start"
        self.output_lines = []
        self.max_output_lines = 10  # Keep last 10 lines
        self.lock = threading.Lock()

    def update(self, current_model: str = None, completed: bool = False, status: str = None, output: str = None):
        with self.lock:
            if current_model:
                self.current_model = current_model
            if completed:
                self.completed += 1
            if status:
                self.status = status
            if output:
                # Split output into lines and add each non-empty line
                new_lines = [line.strip() for line in output.split('\n') if line.strip()]
                self.output_lines.extend(new_lines)
                # Keep only the last max_output_lines
                if len(self.output_lines) > self.max_output_lines:
                    self.output_lines = self.output_lines[-self.max_output_lines:]

def check_existing_judgements(models: List[str], judge: str) -> List[str]:
    """Check which models need to be judged by looking for existing judgement files."""
    to_judge = []
    for model in models:
        needs_judging = False
        for test in TESTS:
            judge_path = Path('data/judgements') / f'judge_{judge}' / test / f'{model}.json'
            if not judge_path.exists():
                needs_judging = True
                break
        if needs_judging:
            to_judge.append(model)
    return to_judge

def run_judge(judge_state: JudgeState, stop_event: threading.Event):
    """Run judgements for a specific judge."""
    while judge_state.remaining_models and not stop_event.is_set():
        model = judge_state.remaining_models[0]
        judge_state.update(current_model=model, status='Running')
        
        start_time = time.time()
        try:
            cmd = [sys.executable, 'judge_answers.py', '-m', model, '-n', '64', '-e', judge_state.name]
            
            # Use Popen to stream output
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            # Stream output in real-time
            while process.poll() is None:
                # Read output line by line
                for line in iter(process.stdout.readline, ''):
                    if line.strip():  # Only update if line is not empty
                        judge_state.update(output=line)
                    if stop_event.is_set():
                        process.terminate()
                        break
            
            if not stop_event.is_set():
                elapsed = time.time() - start_time
                elapsed_str = f'{int(elapsed//60)}min {int(elapsed%60)}s'
                
                if process.returncode == 0:
                    status = f'Last run: Finished in {elapsed_str}'
                else:
                    status = f'{model} judging failed'
                
                judge_state.update(completed=True, status=status)
                judge_state.remaining_models.pop(0)
            
        except Exception as e:
            judge_state.update(status=f'Error: {str(e)}')
            break

def make_progress_panel(judge_state: JudgeState) -> Panel:
    """Create a progress panel for a judge."""
    from rich.console import Group
    from rich.text import Text
    from rich.padding import Padding
    
    # Create progress bar with fixed width ratio
    progress = Progress(
        TextColumn("[bold blue]{task.description:20}"),  # Use string formatting for width
        BarColumn(bar_width=None),  # Allow bar to flex
        TextColumn("{task.fields[fraction]:>10}"),  # Right-aligned, fixed width
        expand=True
    )
    
    task_id = progress.add_task(
        description=f"{judge_state.name}",
        total=judge_state.total_models,
        fraction=f"{judge_state.completed}/{judge_state.total_models}"
    )
    progress.update(task_id, completed=judge_state.completed)
    
    # Create output display with fixed height
    output_lines = []
    with judge_state.lock:
        output_lines = judge_state.output_lines.copy()
    
    # Ensure consistent height by padding or truncating
    target_height = 10
    if len(output_lines) < target_height:
        output_lines.extend([''] * (target_height - len(output_lines)))
    else:
        output_lines = output_lines[-target_height:]
    
    output_text = Text('\n'.join(output_lines), style="bright_black")
    
    # Create a responsive group with padding
    group = Group(
        Padding(progress, (0, 1)),
        Padding(Text(f"Current: {judge_state.current_model}", style="bold yellow"), (1, 1)),
        Padding(Text(judge_state.status, style="green"), (0, 1)),
        Padding(Text("Output:", style="bold blue"), (1, 1)),
        Padding(output_text, (0, 1))
    )
    
    return Panel(group, title=judge_state.name, expand=True)

def main():
    # Read model list from output.csv
    df = pd.read_csv('output.csv')
    models = df['model_name'].unique().tolist()
    
    # Replace '/' with '__' in model names
    models = [model.replace('/', '__') for model in models]
    
    # Check existing judgements for each judge
    judge_states = {}
    for judge in JUDGES:
        remaining_models = check_existing_judgements(models, judge)
        judge_states[judge] = JudgeState(judge, remaining_models)
        print(f"{judge}: {len(remaining_models)}/{len(models)} models to judge")
    
    # Setup TUI with responsive layout
    console = Console()
    layout = Layout(name="root")
    
    # Initialize layout with panels
    panels = {}
    for judge, state in judge_states.items():
        panels[judge] = make_progress_panel(state)
    
    # Create layout sections and add initial content
    layout.split_column(*[Layout(panels[judge], name=judge, ratio=1) for judge in judge_states])
    
    # Setup thread management
    stop_event = threading.Event()
    def signal_handler(signum, frame):
        print("\nStopping all threads...")
        stop_event.set()
    signal.signal(signal.SIGINT, signal_handler)
    
    # Start judge threads with optimized refresh
    with Live(layout, console=console, screen=True, refresh_per_second=10) as live:
        with ThreadPoolExecutor(max_workers=len(JUDGES)) as executor:
            futures = []
            for judge in JUDGES:
                state = judge_states[judge]
                if state.remaining_models:
                    futures.append(
                        executor.submit(run_judge, state, stop_event)
                    )
            
            while futures and not stop_event.is_set():
                # Update each panel individually to reduce flickering
                for judge, state in judge_states.items():
                    panels[judge] = make_progress_panel(state)
                    layout[judge].update(panels[judge])
                
                # Sleep to prevent excessive updates
                time.sleep(0.1)
                
                # Check for completed futures
                futures = [f for f in futures if not f.done()]
                
                # Force a refresh of the display
                live.refresh()
    
    if stop_event.is_set():
        print("\nExecution stopped by user")
    else:
        print("\nAll judgements completed")
    
    if stop_event.is_set():
        print("Gracefully stopped all judge threads")
    else:
        print("All judgements completed")

if __name__ == '__main__':
    main()


"""
# Don't Delete our Docs...

This script should pull all our existing data/answers and judge them with three new judges

- we can read `output.csv` for the model list
- also read `results_vizualization.py` before we start to get a better idea of the naming

The script should first get a list of models

Add to 3 lists:
- llmjudge-tulu405
- llmjudge-llama33
- llmjudge-athenev2

First, for each judge thread:
- Go through each model in the list
- See if model judgement file already exist in data/judgements/$judge_folder, we won't rejudge but skip it
  - You might need to look up how the model answers in data/model_answers are named
  - remove the models from each list

Print out the stats for each model XX/TOTAL to judge for each judge.

Then, kick off a threadpool of 3, each should handle processing 1 judge:

- otherwise run judge script. Here's what the command looks like for each model:
```
python judge_answers.py -m $MODEL -n 64 -e llmjudge-athenev2

python judge_answers.py -m $MODEL -n 64 -e llmjudge-llama33

python judge_answers.py -m $MODEL -n 64 -e llmjudge-tulu405
```

The judge_answers.py has multiple tqdm output and errors that looks like:
```
$ python judge_answers.py -m "ablation-02-llama31-shisa-v2-llama3.1-8b.lr-8e6" -n 64 -e llmjudge-tulu405
/fsx/ubuntu/miniforge3/envs/shaberi/lib/python3.12/site-packages/pydantic/_internal/_config.py:345: UserWarning: Valid config keys have changed in V2:
* 'fields' has been removed
  warnings.warn(message, UserWarning)
Judging ablation-02-llama31-shisa-v2-llama3.1-8b.lr-8e6 on lightblue/tengu_bench using llmjudge-tulu405 (64 proc)
Map (num_proc=64): 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 120/120 [01:01<00:00,  1.94 examples/s]
Creating json from Arrow format: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00, 29.41ba/s]
Judging ablation-02-llama31-shisa-v2-llama3.1-8b.lr-8e6 on elyza/ELYZA-tasks-100 using llmjudge-tulu405 (64 proc)
Map (num_proc=64): 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 100/100 [00:11<00:00,  8.44 examples/s]
Creating json from Arrow format: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00, 62.95ba/s]
Judging ablation-02-llama31-shisa-v2-llama3.1-8b.lr-8e6 on shisa-ai/ja-mt-bench-1shot using llmjudge-tulu405 (64 proc)
num_proc must be <= 60. Reducing num_proc to 60 for dataset of size 60.
Map (num_proc=60): 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 60/60 [00:19<00:00,  3.08 examples/s]
Creating json from Arrow format: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00, 380.30ba/s]
Judging ablation-02-llama31-shisa-v2-llama3.1-8b.lr-8e6 on yuzuai/rakuda-questions using llmjudge-tulu405 (64 proc)
num_proc must be <= 40. Reducing num_proc to 40 for dataset of size 40.
Map (num_proc=40): 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 40/40 [00:27<00:00,  1.46 examples/s]
Creating json from Arrow format: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00, 58.75ba/s]
```
We can log the raw output, we don't really need to see it, but 
It might be nice to have a progress bar for each judge thread like (we can use a TUI lib if it makes it easier to render): 

tulu405
* ##/TOTAL [                 ] IT/s
Last run: Finished in 2min 3s

llama33
* ##/TOTAL [                 ] IT/s
Last run: Finished in 1min 3s

athenev2
* ##/TOTAL [                 ] IT/s
Last run: Finished in 47s
"""