# Shaberi Evaluation Review – Implementation Plan

This document defines how we will systematically review and upgrade our evaluation rubrics, judge prompts, and gold answers so that we **accurately measure current-generation model quality**, especially on cross-lingual tasks like Tengu-Bench.

We will primarily use GPT‑5.1 (and successors) and a small set of other strong models both as:
- **subject-matter helpers** for rubric redesign and bilingual explanation, and  
- **judge models** (after we harden the instructions and rubrics and compare multiple judge configurations).

Use this file as the canonical reference for *how* to run the re-review; use per-item review files (see `SAMPLE-review-item.md`) for *what* we concluded for each question.

---

## 1. Goals and Scope

- **Fix misaligned rubrics and judge prompts**
  - Remove or rewrite criteria that:
    - Reward/penalize properties *not stated* in the user’s instruction.
    - Assume the model can guess hidden intentions (e.g., “should infer that this is a Japanese business email and add a subject line”).
    - Penalize the model for *correctly following* the instruction (e.g., outputting Japanese when asked to translate into Japanese).
- **Make evals robust to strong models**
  - Ensure rubrics do not over-penalize harmless stylistic choices or reasonable paraphrases.
  - Focus scoring on instruction-following, faithfulness, and appropriateness, not on a single “one true style”.
- **Improve cross-lingual transparency**
  - For any item involving Japanese:
    - Provide the *original Japanese* text (marked clearly as **Original (JP)**).
    - Provide an *English explanation/translation* (marked as **Translation (EN)**).
  - This allows non-Japanese-speaking reviewers to meaningfully audit the item.
- **Maintain an auditable record**
  - Every substantial change to an item should have:
    - A per-item review markdown file under `judgement_analysis/` (see naming below).
    - Notes on the original issues, revised instructions, new rubric, and gold answers.

---

## 2. Data We Use Per Item

For each evaluated item, we want to collect the following (from datasets, configs, and logs).  
The goal is to see **how the rubric behaves across a range of model qualities**, not just for a single answer.

- **Metadata**
  - `eval_dataset_key` (e.g., `lightblue/tengu_bench`).
  - `question_id` (bench’s ID, e.g., 18).
  - `category` / subtask label if available (e.g., “Translation EN→JA”).
- **Problem definition**
  - The **original user instruction** (exact text as the model saw).
  - Any system / meta instructions that constrain the task (e.g., “answer in Japanese”).
- **Existing scoring setup**
  - The **rubric / criteria** used by the judge (in original language).
  - English explanation/translation of the rubric, if original is in Japanese.
  - The **judge prompt template** (e.g., as in `evaluation_datasets_config.py`).
  - Example judge outputs (especially for low or surprising scores).
- **Model responses across score ranges**
  - For each item, select a **diverse panel of model answers**, e.g.:
    - A **high-scoring** answer from a strong model.
    - A **medium-scoring** answer from a mid-tier model.
    - A **low-scoring / failure** answer from any model.
    - Optionally, answers from different families (our models + external baselines).
  - For each selected answer, collect:
    - The **raw answer text**.
    - The **judge score**.
    - The **judge reasoning / analysis**, when available.

We will use helper scripts (or small one-off tools) to aggregate this into per-item review docs, so that each question can be assessed with multiple representative answers and judge behaviors.

---

## 3. LLM-Assisted Review Workflow (Per Item)

### Step 1 – Item Extraction

For a given `(dataset, question_id)`:

1. Programmatically gather:
   - Question / instruction text.
   - Current rubric / criteria.
   - Judge prompt template.
   - A panel of model answers spanning high / medium / low scores (see Section 2).
   - Existing judge outputs (score + explanation) for those answers.
2. Normalize language:
   - If the question, rubric, or answer is in Japanese:
     - Include **Original (JP)** text.
     - Ask GPT‑5.1 for a concise **Translation (EN)** aimed at reviewers, *not* as a gold answer.

This content becomes the “raw context” section of the per-item review file.

### Step 2 – Triage: Is the Item Fundamentally Broken?

Using GPT‑5.1 as an assistant (not as the judge yet), we ask it to:

- Compare:
  - the **user instruction**,
  - the **rubric / criteria**, and
  - one or more **model answers + judge outputs**.
- Label the item as one of:
  - **A – OK**: Rubric and judge behavior are broadly consistent with the instruction. Only minor clarifications needed.
  - **B – Needs adjustment**: Same task is fine, but rubric or judge prompt needs non-trivial edits (weights, wording, edge cases).
  - **C – Mis-specified**: Rubric and/or judge are evaluating a *different task* than the instruction (e.g., Tengu Q18 case).

We will prioritize **B** and especially **C** items for full rework.

### Step 3 – Rubric Redesign with GPT‑5.1

For items labeled **B** or **C**:

1. Provide GPT‑5.1 with:
   - Instruction, rubric, judge prompt, and model answers.
   - A short description of the issues (e.g., “rubric rewards English email quality, but the instruction is EN→JA translation”).
2. Ask GPT‑5.1 to:
   - Propose a **new rubric** that:
     - Evaluates the **task actually stated in the instruction**.
     - Uses a clear point structure (e.g., sums to 10).
     - Specifies what *not* to penalize (e.g., reasonable choice of `さん` vs `様`; absence of subject line if not in source).
   - Provide the rubric in:
     - **Japanese**, tuned for the judge prompt.
     - **English**, for our internal documentation.

The human reviewers will refine and approve this rubric.

### Step 4 – Judge Prompt Redesign

Once we have a revised rubric:

1. Update the **judge prompt template** (e.g., in `evaluation_datasets_config.py`) so that it:
   - Explicitly describes the task (e.g., “evaluate EN→JA email translation quality”).
   - References the new rubric verbatim.
   - Clarifies expectations about languages:
     - When translation to Japanese is requested, **do not** penalize the answer for not being in English.
   - Clarifies what *not* to punish (e.g., stylistic differences that do not change meaning).
2. Ensure output format is robust:
   - Always require `FINAL SCORE: x` on the last line.
   - Keep judge explanations but do not depend on them for parsing.

### Step 5 – Gold Answers and Calibration Examples

For each reworked item:

1. Ask GPT‑5.1 to produce:
   - One or more **gold answers** that fully meet the rubric.
   - Optionally, some **borderline** answers (e.g., minimal translations) and clearly **bad** answers.
2. For cross-lingual tasks:
   - Provide:
     - **Gold Output (JP)** – what the model *should* roughly produce.
     - **Back-translation (EN)** – for reviewers to confirm the meaning.
   - Clearly mark which is the original source vs translation vs back-translation.

These examples help both:
- Human reviewers (to sanity-check the task), and
- The LLM judge (through the “few-shot” part of its prompt, if we add examples).

### Step 6 – Human Expert Pass

Each reworked item should be signed off by:

- At least one **Japanese-fluent reviewer**:
  - Checks nuance, politeness, discourse structure, and whether the rubric captures key linguistic subtleties.
- At least one **non-Japanese-speaking reviewer**:
  - Ensures the English documentation and rubric explanation are clear enough to audit without Japanese skills.

They will:

- Use the per-item review file (see `SAMPLE-review-item.md`).
- Mark a simple checklist: instruction/rubric alignment, gold answers OK, judge prompt OK, etc.

### Step 7 – Implementation and Rejudging

Once an item’s review is approved:

1. **Update code / configs**
   - Modify the relevant evaluation configuration (e.g., Tengu prompt builder) to use the new rubric and judge prompt.
2. **Re-run the judge**
   - Recompute scores for the affected dataset(s) with the updated judge setup.
   - Keep both **old** and **new** scores for comparison and sanity checks.
3. **Document changes**
   - In the per-item review file:
     - Record the date, judge model version, and a short change log.
   - Optionally maintain a summary table of reworked items in this file or a separate index.

---

## 4. File Naming and Organization

- Per-item review files live under `judgement_analysis/`.
- Suggested naming:
  - `judgement_analysis/<dataset>__q<id>__<short-task-label>.md`
  - Example: `judgement_analysis/lightblue__tengu_bench__q18__en-ja-email-translation.md`
- For exploratory or scratch notes:
  - Use `judgement_analysis/notes/` or explicitly mark sections as “Scratch” vs “Final”.

`SAMPLE-review-item.md` provides a concrete example (Tengu Q18) of how to structure these files.

---

## 5. Tengu Q18 as Motivating Example

For quick reference, the specific failure that motivated this review process:

- **Instruction**: “Translate the following to Japanese. ### Email …” (English email to Japanese).
- **Original rubric** (English paraphrase):
  - Considers differences in email conventions between Japanese and English-speaking cultures: 3 points
  - Considers differences between Japanese and English (Japanese places the conclusion at the end): 3 points
  - Translates the source text completely and accurately: 3 points
  - Uses natural English: 1 point
- **Observed problems**:
  - The rubric implicitly expects an *English* email-style rewrite and penalizes the model for outputting Japanese, even though the task is to **translate into Japanese**.
  - It expects the model to infer extra constraints (e.g., add a subject line, upgrade `田中さん` to `田中様`, reorder content to match stereotypical Japanese email structure) that are **not in the instruction**.

`SAMPLE-review-item.md` shows how we will document and repair issues like this systematically.

---

## 6. Dataset-Level Coverage and Thoroughness

While Section 3 describes a per-item workflow, we also need a **dataset-level plan** to ensure coverage and appropriate rating across each eval:

- **6.1 Map each eval type**
  - Classify each dataset by task type (e.g., translation, summarization, QA, reasoning, safety) and languages involved.
  - For each class, define shared expectations (e.g., translation faithfulness vs localization, allowed stylistic freedom).

- **6.2 Systematic question sampling**
  - For large evals, review:
    - A **random sample** of items.
    - A **stratified sample** over categories / difficulty tags if available.
    - Items with **largest disagreements** between models or judges (if multiple judge runs exist).
  - For smaller evals (like many Japanese benchmarks), aim for **full question coverage** over time.

- **6.3 Score-distribution–aware review**
  - For each dataset, compute and inspect:
    - Score histograms per model.
    - Items that frequently produce **very low** or **very high** scores across many models.
  - Prioritize:
    - Items where strong models receive inexplicably low scores.
    - Items where weak models routinely score unreasonably high.

- **6.4 Cross-model sanity checks**
  - For a given question, confirm that:
    - Better answers (by human judgment) get **higher scores** on average.
    - The rubric does not systematically prefer one model’s quirks (e.g., overly verbose style) over another when content quality is similar.

These dataset-level checks tell us whether we are “appropriately rating” models on that eval as a whole, beyond any individual question.

---

## 7. Handling Missing Judge Reasoning

In some existing runs, only the **numeric score** was stored, without the judge’s textual reasoning. For thorough review:

- **7.1 When judge reasoning is missing**
  - If we still have:
    - The **model answer**, and
    - The **judge prompt / rubric** used at the time,
  - Then we can:
    - Re-run the judge with the same or updated model to obtain **fresh reasoning** aligned with the current rubric.
    - Or, run GPT‑5.1 in **“post-hoc analyst” mode**:
      - Provide the question, rubric, and model answer.
      - Ask for a narrative evaluation and an approximate score, then compare to the stored score.

- **7.2 When we only have scores**
  - If answers themselves are not stored and we only have scores:
    - Treat these as **legacy metrics** that cannot be deeply audited.
    - Avoid using them as ground truth during rubric redesign.
    - Once the new rubric and judge prompt are ready, **re-run** the eval from model answers (if we can regenerate or still have them).

- **7.3 Preference going forward**
  - For every future judge run:
    - Store **both**:
      - The **numeric score**.
      - The **full judge output** (reasoning, selected criteria, calculation).
    - Ensure file formats (e.g., under `data/judgements/`) always include a `judge_output` or equivalent field.
  - This enables:
    - Post-hoc audits of individual items.
    - Training of improved judges or heuristics from real judge rationales.

These practices make the review process repeatable and diagnosable for future versions of the evals.

---

## 8. Current Dataset Scale (Snapshot)

This section summarizes how many **items** and **stored judgements** we currently have per eval, based on `data/model_answers/` and `data/judgements/`.

- **Datasets (items per eval)**
  - `elyza__ELYZA-tasks-100`: **100 items**
  - `lightblue__tengu_bench`: **120 items**
  - `shisa-ai__ja-mt-bench-1shot`: **60 items**
  - `yuzuai__rakuda-questions`: **40 items**
  - Legacy: `lightblue__japanes-mt-bench-oneshot`: **80 items** (model answers exist; no judgements, superseded by `shisa-ai__ja-mt-bench-1shot`)

- **Judgement counts per dataset (all judges combined)**
  - `elyza__ELYZA-tasks-100` – **152,500** judgements  
    - `judge_gpt-4-turbo-preview`: 14,100  
    - `judge_gpt-4.1-2025-04-14`: 16,900  
    - `judge_gpt-5.1-2025-11-13`: 7,700  
    - `judge_gpt-5.1-2025-11-13-oldjudgeprompt`: 100  
    - `judge_llmjudge-athenev2`: 37,900  
    - `judge_llmjudge-llama33`: 37,900  
    - `judge_llmjudge-tulu405`: 37,900  
  - `lightblue__tengu_bench` – **183,000** judgements  
    - `judge_gpt-4-turbo-preview`: 16,920  
    - `judge_gpt-4.1-2025-04-14`: 20,280  
    - `judge_gpt-5.1-2025-11-13`: 9,240  
    - `judge_gpt-5.1-2025-11-13-oldjudgeprompt`: 120  
    - `judge_llmjudge-athenev2`: 45,480  
    - `judge_llmjudge-llama33`: 45,480  
    - `judge_llmjudge-tulu405`: 45,480  
  - `shisa-ai__ja-mt-bench-1shot` – **91,680** judgements  
    - `judge_gpt-4-turbo-preview`: 8,640  
    - `judge_gpt-4.1-2025-04-14`: 10,140  
    - `judge_gpt-5.1-2025-11-13`: 4,620  
    - `judge_gpt-5.1-2025-11-13-oldjudgeprompt`: 60  
    - `judge_llmjudge-athenev2`: 22,740  
    - `judge_llmjudge-llama33`: 22,740  
    - `judge_llmjudge-tulu405`: 22,740  
  - `yuzuai__rakuda-questions` – **61,000** judgements  
    - `judge_gpt-4-turbo-preview`: 5,640  
    - `judge_gpt-4.1-2025-04-14`: 6,760  
    - `judge_gpt-5.1-2025-11-13`: 3,080  
    - `judge_gpt-5.1-2025-11-13-oldjudgeprompt`: 40  
    - `judge_llmjudge-athenev2`: 15,160  
    - `judge_llmjudge-llama33`: 15,160  
    - `judge_llmjudge-tulu405`: 15,160  

- **Overall scale**
  - Across these evals, we have **400 unique items** and roughly **488,180 stored judgements** (item × model × judge runs).  
  - Any full re-review will therefore focus on:
    - **Per-item rubric correctness** (400 questions), and  
    - **Sampling** representative model answers and judge outputs rather than manually reading all ~0.5M judgements.

---

## 9. New `shisa-ai/shaberi-v3-[eval]` Datasets

As we re-review the existing evals, we will create our own **normalized v3 datasets** under a new namespace, and treat them as the canonical source of truth going forward.

- **9.1 Target source evals for re-review**
  - `elyza__ELYZA-tasks-100`
  - `lightblue__tengu_bench`
  - `shisa-ai__ja-mt-bench-1shot` (replaces legacy `lightblue__japanes-mt-bench-oneshot`)
  - `yuzuai__rakuda-questions`

- **9.2 New dataset naming**
  - New datasets will live under a `shisa-ai/shaberi-v3-[eval]` naming scheme, for example:
    - `shisa-ai/shaberi-v3-tengu`
    - `shisa-ai/shaberi-v3-ja-mt-bench`
    - `shisa-ai/shaberi-v3-elyza-tasks-100`
    - `shisa-ai/shaberi-v3-rakuda-questions`
  - Each v3 dataset will:
    - Be derived from a legacy source eval plus our review decisions.
    - Clearly track the mapping from original dataset and question ID to the new item ID.

- **9.3 Normalized item schema**
  - Every item in the v3 datasets should include at least:
    - `id` / `question_id`
    - `source_dataset` and `source_question_id` (legacy provenance)
    - `question_original` (e.g., `question_jp` or `question_en`)
    - `question_translation` where applicable (e.g., JP↔EN for reviewers)
    - `task_type` (e.g., `en-ja-translation`, `qa`, `summarization`)
    - `rubric_jp` (or primary rubric language)
    - `rubric_en` (English explanation for reviewers)
    - `gold_answer` (primary gold output, possibly in Japanese)
    - `gold_answer_backtranslation_en` for cross-lingual tasks
    - Optional: category / difficulty / tags
  - Items that are judged unsalvageable (e.g., fundamentally mis-specified) will be:
    - Either omitted from the v3 dataset entirely, or
    - Included with a flag such as `excluded: true` and an explanation.

- **9.4 Policy for removing / modifying items**
  - During review we may:
    - **Refine** instructions and rubrics to better reflect the intended task.
    - **Split** ambiguous items into clearer variants (if needed).
    - **Remove** items that cannot be made coherent even with rubric fixes.
  - The v3 datasets will only contain items that:
    - Have aligned instruction + rubric + gold answer.
    - Passed the dual human review (Japanese-fluent + non-Japanese-speaking reviewer).

- **9.5 Use of v3 datasets**
  - All future:
    - Model evaluation runs,
    - Judge benchmarking,
    - Comparative studies between judges,
  - should use `shisa-ai/shaberi-v3-[eval]` as the primary datasets. Legacy datasets remain for reference and back-comparisons only.

---

## 10. Judge Model Comparison, Prompt Language, and Cost

To ensure our revised rubrics and judge prompts work robustly and cost-effectively, we will **compare several judge models**, test **English vs Japanese prompts**, and track **token usage**.

- **10.1 Judge models to test**
  - **Hosted / API models**
    - GPT‑5.1 – low reasoning and no reasoning modes
    - GPT‑5.1 mini – low reasoning and no reasoning
    - Gemini 3 Pro – low reasoning
    - Gemini 2.5 Flash – low reasoning and no reasoning
  - **Local models**
    - `gpt-oss-120b`
    - `Shisa V2 405B`
    - `Gemma 3 27B`

- **10.2 Prompt language experiments (EN vs JP)**
  - For each dataset and task type, we will test at least:
    - **Japanese-only** judge prompts (current default for many JP evals).
    - **English-only** judge prompts.
    - **Bilingual** prompts (e.g., instructions in English plus JP rubric text).
  - Motivation:
    - Many models show stronger **instruction-following** in English.
    - Some subtleties in Japanese tasks may be easier to express in Japanese.
  - For each judge model + prompt language configuration, we will:
    - Evaluate a **shared subset of items** with human-reviewed ground truth.
    - Measure:
      - Alignment with human scores.
      - Stability across models (e.g., whether better answers consistently score higher).

- **10.3 Reasoning mode experiments**
  - For models that support it (e.g., GPT‑5.1, GPT‑5.1 mini, Gemini 2.5 Flash):
    - Compare:
      - **Low reasoning**: brief justification + score.
      - **No reasoning**: directly output a score with minimal explanation.
  - We will:
    - Check whether reasoning improves **consistency** and **alignment** with human judgments.
    - Decide where reasoning is worth the extra token/cost overhead (e.g., for calibration runs vs large-scale routine scoring).

- **10.4 Token counting and cost tracking**
  - For each judge configuration (model × prompt language × reasoning mode), we will track:
    - Average **prompt tokens** per evaluation.
    - Average **completion tokens** per evaluation.
    - Total tokens and **estimated cost** for a full pass over each v3 dataset.
  - For local models:
    - Track approximate **compute cost** (e.g., wall-clock time, GPU-hours) instead of API cost.
  - This data will be recorded alongside evaluation results so we can:
    - Choose a **default judge** for production runs that balances quality and cost.
    - Reserve more expensive configurations (e.g., GPT‑5.1 with low reasoning) for:
      - Calibration,
      - Spot checks,
      - High-stakes comparisons between models.

- **10.5 Final judge selection**
  - After experiments, we will:
    - Pick one or two **primary judge configurations** per dataset class (e.g., translation vs QA).
    - Document:
      - Which judge and prompt language is used as the **main metric**.
      - Which alternatives are used for robustness checks.
  - All choices should be justified in terms of:
    - Alignment with human evaluations,
    - Stability across models,
    - Acceptable token/compute cost.
