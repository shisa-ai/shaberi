
  | Version | ELYZA | JA MT | Rakuda | Tengu |
  |---------|------:|------:|-------:|------:|
  | 4.1 Old | 4.33  | 9.50  |   9.55 |  7.65 |
  | 5.1 Old | 3.82  | 7.55  |   7.29 |  6.60 |
  | 5.1 New | 4.04  | 7.53  |   7.70 |  6.44 |

`5.1 Old` corresponds to the directory `judge_gpt-5.1-2025-11-13-oldjudgeprompt/`, where GPT-5.1 used the older, looser judge prompts.  
`5.1 New` corresponds to `judge_gpt-5.1-2025-11-13/`, where GPT-5.1 uses the updated prompts and score-format instructions described below.  
`4.1 Old` corresponds to `judge_gpt-4.1-2025-04-14/`.

Overall, GPT‑5.1 is already stricter and more conservative than GPT‑4.1, so even the “old” 5.1 prompts tend to assign lower scores than 4.1 for the same model outputs. The updated 5.1 prompts therefore only shift the averages modestly; most of the difference between 4.1 vs 5.1 comes from the judge model itself rather than the prompt tweaks.

## Prompt changes (5.1 Old → 5.1 New)

- All Japanese judge prompts (Tengu, ELYZA, JA MT, Rakuda) were revised to:
  - Emphasize that the answer must be in correct, natural Japanese.
  - Strongly penalize non‑Japanese answers, inappropriate language, nonsensical output, and repetitive/looping text.
  - Make the role of the judge clearer (客観的・公平な評価者) and explicitly tie the score to usefulness, correctness, relevance, and detail.
- Tengu:
  - Restored `[正解例]` as the label for the example answer (instead of `[評価]`), and clarified that the final score is a 0–10 numeric rating based on the listed criteria.
- ELYZA:
  - Added explicit “basic deduction items” for bad Japanese, factual errors, over‑cautious safety refusals, non‑Japanese output, and nonsensical/looping output.
- MT‑Bench & Rakuda:
  - Updated instructions to highlight Japanese fluency, penalize incoherent or looping responses, and reinforce that scores should reflect overall answer quality, not just partial correctness.

As the averages above show, these changes make 5.1’s behavior a bit stricter in the right places, but the dominant effect is the difference between GPT‑4.1 vs GPT‑5.1 as judges rather than the prompt wording alone.

## Score formatting and parsing changes

We also changed how the judge is asked to emit scores and how we parse them, to reduce failures and weird edge cases:

- Unified final‑score format:
  - All judge prompts now instruct the model to put the final numeric score on the last line in the form  
    `FINAL SCORE: x`  
    where `x` can be an integer or decimal (e.g., `8` or `8.5`).
  - Prompts explicitly say not to write `FINAL SCORE:` on any other line.
- Parsing logic:
  - For each evaluation, we parse all occurrences of `FINAL SCORE:\s*([0-9.]+)` in the judge output and **use the last one**, which is robust even if the model mistakenly emits multiple scores during its reasoning.
  - We still accept decimal scores and round with `round(float(x))` to match the 0–10 or 1–5 scales.
  - For backward compatibility, evaluators fall back to the previous formats when `FINAL SCORE` is missing:
    - `<score>number</score>` XML‑style tags.
    - Legacy `評価：[[5]]` style strings.
    - In Rakuda, any `X/10` style ratings are detected and averaged as a last resort.
- Tengu and ELYZA evaluators:
  - Handle the new `FINAL SCORE` format first, but still know how to parse older judge outputs generated before this change.

This combination (stronger instructions + a single, consistent `FINAL SCORE` format with robust regex parsing) is intended to make judge behavior more stable across runs and models while minimizing score‑parsing failures.  
