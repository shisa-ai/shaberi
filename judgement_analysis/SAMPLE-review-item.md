# SAMPLE Review Item – Tengu-Bench Q18 (EN→JA Email Translation)

This file is a **template plus concrete example** of how to document a single item review. Replace dataset/IDs/content as needed for other items, but keep the overall structure.

---

## 1. Metadata

- **Dataset**: `lightblue/tengu_bench`
- **Question ID**: `18` (Tengu-Bench internal ID)
- **Model under focus**: `183` (internal model code; replace as needed)
- **Task type**: English → Japanese email translation
- **Current judge model**: GPT‑5.1 (or as configured at run time)
- **Status**: Draft / Example (not yet wired into code)

---

## 2. Original Task and Text

### 2.1 Instruction Shown to the Model

**Original (EN – User instruction and source email)**:

> Translate the following to Japanese.  
>  
> ### Email  
>  
> Hey Tanaka-san,  
>  
> Thank you for your help last week with or proposal to the government. You really knocked it out of the park.  
>  
> Can you send me the results from the experiments from the past 2 months? I have a meeting with Bridgette and Fumi at noon and need to explain what we've been doing.  
>  
> Regards,  
>  
> Jo

Interpretation: The model is asked **only** to translate this email into Japanese. There is no instruction to:
- Re-design the email,
- Add a subject line,
- Change the level of politeness, or
- Reorder the content beyond what is needed for a natural translation.

---

## 3. Existing Rubric / Judge Behavior (Problematic)

### 3.1 Original Criteria (English paraphrase)

The criteria in use at the time of the anomaly were (paraphrased from logs):

- Considers the differences in email writing conventions between Japanese and English-speaking cultures: **3 points**
- Considers the differences between Japanese and English (Japanese places the conclusion at the end): **3 points**
- Translates the source text completely and accurately, without omissions or additions: **3 points**
- Uses natural English: **1 point**

### 3.2 Observed Issues

From team discussion and judge outputs:

- The model correctly produced a **Japanese email**, but:
  - Lost points because the output “was not in English” under the “Uses natural English” criterion.
- The judge expected:
  - Use of `〜様` instead of `〜さん` (e.g., “Tanaka-sama” vs “Tanaka-san”), even though this is not specified.
  - Inclusion of an email subject line, even though none appears in the source.
  - Reordering of sentences to match a particular stereotype of Japanese business email structure.
- **Net effect**:
  - The rubric and judge were effectively grading a different task: *“Write an appropriate Japanese-style business email in English and/or Japanese, incorporating inferred cultural conventions”* rather than *“Translate the given email into Japanese.”*

This is a canonical example of a **C – Mis-specified** item (see `IMPLEMENTATION-review-eval.md`).

---

## 4. Issue Analysis

### 4.1 Instruction–Rubric Mismatch

- Instruction: pure **EN→JA translation**.
- Rubric: partially evaluates:
  - Cultural adaptation choices that were never required.
  - Email format features (subject line) not present in the source.
  - **English** fluency for an output that is supposed to be **Japanese**.

This leads to:

- Penalizing correct translations that:
  - Do not add a subject line.
  - Use reasonable but not hyper-formal honorifics (e.g., `田中さん`).
  - Preserve source order where still natural in Japanese.
- Rewarding or expecting behavior that **goes beyond** the stated task and may actually be undesirable in a strict translation setting (e.g., adding content not present in the source).

### 4.2 Consequences for Strong Models

As models improve:

- They often make *reasonable stylistic choices* (e.g., choosing a slightly different but natural email structure).
- The current rubric is brittle and can:
  - Penalize high-quality translations for not matching a very specific, unspoken template.
  - Under-reward faithful translations that avoid unnecessary additions.

Hence the need to **rewrite the rubric and judge prompt** so they score the actual task.

---

## 5. Proposed Revised Task & Rubric

### 5.1 Task Restatement (What We Actually Want)

- **Task**: Translate the given English email into natural Japanese.
- **Primary goal**: Preserve the meaning and tone of the original email as a casual–businesslike message between colleagues.
- **Allowed adaptations**:
  - Minor adjustments to fit Japanese email norms (e.g., greeting format, slight reordering for natural flow).
  - Reasonable choice of honorifics (`さん` vs `様`) as long as the tone is broadly appropriate.
- **Not required / not mandatory**:
  - Adding a subject line.
  - Upgrading honorifics beyond what is implied in the source.
  - Major content additions or omissions.

### 5.2 Revised Rubric (Japanese – for Judge Prompt)

**Revised criteria (total 10 points):**

- **意味の正確さ・完全性: 4点**
  - 原文の情報（依頼内容、感謝の内容、登場人物、日時など）が漏れなく日本語に反映されている。  
  - 重要な意味の抜けや追加・歪曲がない。
- **トーンと丁寧さの適切さ: 3点**
  - 英文の雰囲気（ややカジュアルだがビジネス寄り）に近い丁寧さで訳されている。  
  - 「田中さん」「田中様」などの選択は文脈上不自然でなければ減点しない。
- **日本語としての自然さ・文法: 3点**
  - 日本語として自然で読みやすく、重大な文法誤りがない。  
  - 必要に応じた語順の調整や文の分割・結合は許容する。

**減点しないことを明示する注意点:**

- 件名（Subject）が原文にない場合、件名を追加していなくてもそれだけで減点しない。  
- 原文にない情報（会社名、部署名など）を過度に付け加えて意味を変えていない限り、軽微な補足は大きく減点しない。  
- 「日本語のメールとしての一般的な定型」に完全に一致しないことのみを理由に、大幅な減点を行わない。

### 5.3 Rubric Explanation (English – for Reviewers)

- **Faithfulness & completeness (4 pts)**
  - All key information from the source (thanks for help, government proposal, two months of experiment results, meeting with Bridgette and Fumi at noon) appears in the Japanese email.
  - No major meaning changes, omissions, or unjustified additions.
- **Tone & politeness (3 pts)**
  - The politeness level roughly matches a polite but not hyper-formal business email between colleagues.
  - Choices like `田中さん` vs `田中様` are acceptable as long as they are not clearly inappropriate.
- **Natural Japanese & grammar (3 pts)**
  - The Japanese is grammatically correct and reads naturally as an email.
  - Reasonable reordering and sentence splitting/merging are allowed.
- **Do not penalize solely for**:
  - Missing subject line, since none exists in the source.
  - Minor stylistic differences that do not change meaning.

---

## 6. Gold Answer and Back-Translation

### 6.1 Gold Output (JP – Target Translation)

**Gold Answer (JP – model output target, not shown to models):**

> 田中さん  
>  
> 先週は、政府への提案書の件でご協力いただきありがとうございました。おかげさまで、とても良い内容に仕上がりました。  
>  
> この2か月間の実験結果を送っていただけますか。正午からブリジットさんとフミさんとの打ち合わせがあり、これまでの取り組みについて説明する必要があります。  
>  
> よろしくお願いします。  
>  
> Jo

Notes:
- Uses `田中さん` (not `田中様`), which we treat as acceptable.
- Slightly paraphrases “you really knocked it out of the park” into a natural compliment.

### 6.2 Back-Translation (EN – For Reviewers Only)

**Back-translation (EN – for human reviewers; not used as a reference answer):**

> Tanaka-san,  
>  
> Thank you very much for your help last week with the proposal to the government. Thanks to you, it turned out very well.  
>  
> Could you send me the results of the experiments from the past two months? I have a meeting with Bridgette and Fumi at noon and need to explain what we have been doing.  
>  
> Best regards,  
> Jo

This shows that:
- Meaning and key details are preserved.
- Tone is polite and appropriate.

---

## 7. Revised Judge Prompt Sketch

This is a **sketch** of how the judge prompt for this item type (EN→JA email translation) should be structured. The actual implementation will be integrated into `evaluation_datasets_config.py`.

**Prompt skeleton (Japanese):**

> [指示]  
> あなたは翻訳品質の評価者です。以下の英語のメールを日本語に翻訳した「評価するモデルの回答」を、採点基準に従って0〜10点で評価してください。  
>  
> 回答は正しく自然な日本語であることが望ましいです。日本語でない場合や、意味が大きく異なる場合は大きく減点してください。  
>  
> ただし、件名を追加していないことや、「田中さん」「田中様」などの敬称の違いなど、意味を変えない範囲のスタイルの違いだけを理由に大幅な減点をしないでください。  
>  
> [質問]  
> （英語のメール本文）  
>  
> [評価項目]  
> （上記の新しい採点基準をそのまま挿入）  
>  
> [評価するモデルの回答]  
> （モデルの日本語メール）  
>  
> # 以下の形式で回答してください。最後の行に必ず「FINAL SCORE: x」（x は最終スコアの数値）だけを出力してください。  
> [該当する評価項目とその簡潔な理由]  
>  
> [計算式]  
>  
> [点数]  
>  
> FINAL SCORE: #

This keeps the Tengu-style format but aligns the criteria with the actual task.

---

## 8. Human Review Checklist (to be filled when finalizing)

- [ ] Instruction and rubric describe the **same task**.  
- [ ] Revised rubric is clear in both Japanese and English.  
- [ ] Gold JP answer reviewed by a Japanese-fluent reviewer.  
- [ ] Back-translation reviewed by a non-Japanese-speaking reviewer for clarity.  
- [ ] Judge prompt text integrated into `evaluation_datasets_config.py` (or relevant config).  
- [ ] Item rejudged and scores compared against previous version.  
- [ ] Final status updated (e.g., “Approved – New rubric in production”).  

When using this file as a template, copy it, update the metadata, replace the texts and rubrics, and then walk through the checklist before marking the item as complete.

