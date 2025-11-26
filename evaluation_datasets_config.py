import re
from llm_functions import get_model_response


def _shorten_for_log(value, max_len: int = 80) -> str:
    """Utility: collapse newlines and truncate long strings for logging."""
    if value is None:
        return ""
    s = str(value).replace("\n", " ").replace("\r", " ")
    return (s[: max_len - 3] + "...") if len(s) > max_len else s


def _extract_question_id(data: dict) -> str:
    """
    Try to extract a compact identifier for the sample so that logs are
    legible without dumping entire prompts/responses.
    """
    for key in ("id", "ID", "index", "Index", "Category", "Question", "text", "input"):
        if key in data:
            return _shorten_for_log(data.get(key), 40)
    return ""


def _extract_answer_snippet(data: dict) -> str:
    """
    Try to pull a short snippet of the model's original answer for logging.
    """
    for key in ("ModelAnswer", "Answer", "output", "response", "answer"):
        if key in data:
            return _shorten_for_log(data.get(key), 60)
    return ""


def _log_no_score(source: str, data: dict, evaluation_text: str | None, note: str = "") -> None:
    """
    Compact, production-friendly logging when no numeric score can be
    parsed, instead of printing entire LLM responses.
    """
    qid = _extract_question_id(data or {})
    eval_head = _shorten_for_log(evaluation_text or "", 80)
    answer_head = _extract_answer_snippet(data or {})
    eval_len = len(evaluation_text) if evaluation_text is not None else 0
    parts: list[str] = [f"NO SCORE FOUND [{source}]"]
    if qid:
        parts.append(f"id='{qid}'")
    if answer_head:
        parts.append(f"answer_head='{answer_head}'")
    if eval_head:
        parts.append(f"eval_head='{eval_head}'")
    parts.append(f"eval_len={eval_len}")
    if note:
        parts.append(note)
    print(" ".join(parts))


def _fallback_score_from_analysis(
    analysis_text: str,
    data: dict,
    judge_model_name: str,
    scale_description: str,
) -> int | None:
    """
    Generic guard: if the primary evaluation output does not contain a
    parseable FINAL SCORE, ask the judge once more to extract just the
    numeric score from its own analysis.
    """
    if analysis_text is None:
        print("Fallback score requested but analysis_text is None; returning None.")
        return None

    analysis_text = str(analysis_text).strip()
    if not analysis_text:
        print("Fallback score requested but analysis_text is empty; returning None.")
        return None

    # Include original question and answer for maximum robustness when backfilling.
    question_text = ""
    for key in ("Question", "text", "input", "prompt"):
        if key in (data or {}):
            question_text = str((data or {}).get(key) or "")
            break
    answer_text = ""
    for key in ("ModelAnswer", "Answer", "output", "response", "answer"):
        if key in (data or {}):
            answer_text = str((data or {}).get(key) or "")
            break

    fallback_prompt = f"""We are evaluating a model response. The original question, the model's answer, and an existing judge evaluation are given below. The judge appears to have forgotten to include the numeric FINAL SCORE.

Please carefully read the question, the model answer, and the judge evaluation. Then output ONLY a single line in the exact format:

FINAL SCORE: [#]

where [#] is the final numeric score {scale_description}.

[Question]
{question_text}

[Model answer]
{answer_text}

[Judge evaluation]
{analysis_text}

*** 
IMPORTANT REMINDER: Do not explain. Do not add any other text or punctuation. Output exactly one line 'FINAL SCORE: [#]'.
"""

    messages = [{"role": "user", "content": fallback_prompt}]

    try:
        fallback_output = get_model_response(messages, judge_model_name)
    except Exception as e:
        print(f"Error while calling fallback score prompt with judge {judge_model_name}: {e}")
        return None

    try:
        # Primary: look for FINAL SCORE pattern
        matches = re.findall(r"FINAL SCORE:\s*([0-9.]+)", str(fallback_output))
        if matches:
            return round(float(matches[-1]))

        # Fallback: treat the whole output as a bare number
        return round(float(str(fallback_output).strip()))
    except (ValueError, AttributeError):
        _log_no_score("FALLBACK", data or {}, fallback_output, note="fallback_parse_error")
        return None

######### TENGU ##########

tengu_example_question_answer = {
    "Question": "「急がば回れ」という言葉について説明してください。",
    "Answer": "「急がば回れ」という言葉は、日本の諺の一つであり、直接的な意味は「急ぐときは、早道や危険な方法を選ばずに、むしろ回り道で確実で安全な道を通った方が結局は早く着けるものだ」というものです。この言葉は、物事は慌てずに着実に進めることが結果としてうまくいくという教訓を含んでいます",
    "Criteria": "- 本来の「急ぐときは、早道や危険な方法を選ばずに、むしろ回り道で確実で安全な道を通った方が結局は早く着ける」という意味について説明している:3点\n- 一般化した「物事は慌てずに着実に進めることが結果としてうまくいく」という意味について説明している:3点\n- ことわざであることを示している:2点\n- 説明は具体的でわかりやすい:1点\n- 自然な日本語である:1点",
    "ModelAnswer": "「急がば回れ」とは、物事を急いで進めるよりも、慎重に計画を立てて行動する方が結果が良くなるという意味のことわざです。つまり、無駄なミスやトラブルを避けるためには、急いで手を打つのではなく、ゆっくりと計画を練り、周囲をよく考えて行動することが大切だということを教えています。急いで物事を進めようとして失敗してしまうよりも、手間と時間をかけてじっくりと準備をする方が結果的に効率的で成功する可能性が高いという教訓を持つ言葉です。"
}

tengu_example_evaluation = """[該当する評価項目とその簡潔な理由]
- 一般化した「物事は慌てずに着実に進めることが結果としてうまくいく」という意味について説明している:3点
  - 「物事を急いで進めるよりも、慎重に計画を立てて行動する方が結果が良くなるという意味」と示している。
- ことわざであることを示している:2点
  - 「ことわざです」と書いてある。
- 説明は具体的でわかりやすい:1点
  - 言い換えたりしながら詳しく説明されている。
- 自然な日本語である:1点
  - 日本語の用法が全て正しい。
[計算式]
3+2+1+1=7
[点数]
<score>7</score>"""

def get_tengu_prompt(data: dict) -> str:
    question = data['Question']
    example_answer = data['Answer']
    criteria = data['Criteria']
    model_answer = data['ModelAnswer']

    answer_bool = example_answer is not None
    
    example_answer = f"\n[正解例]\n{example_answer}\n" if answer_bool else ""
    answer_explanation = "'正解例'," if answer_bool else ""
    
    prompt = f"""[指示]
あなたは評価者です。以下の'質問',{answer_explanation}'評価項目'に基づいて'評価するモデルの回答'を0~10点の数値で評価してください。客観的かつ公平に評価してください。回答は正しく自然な日本語であるべきです。日本語でない場合や不適切な言語表現、支離滅裂・意味不明な出力、反復/ループがある場合は大きく減点してください。

[質問]
{question}
{example_answer}
[評価項目]
{criteria}

[評価するモデルの回答]
{model_answer}

# 以下の形式で回答してください。最後の行に必ず 'FINAL SCORE: x' （x は最終スコアの数値）だけを出力してください。
[該当する評価項目とその簡潔な理由]

[計算式]

[点数]（途中経過を示してもよい）

FINAL SCORE: #
"""
    return prompt

def get_tengu_eval_score(eval_text: str) -> int | None:
    if eval_text is None:
        print("Received None eval_text, returning None score.")
        return None
    try:
        # Prefer FINAL SCORE pattern, using the last occurrence if multiple appear
        matches = re.findall(r"FINAL SCORE:\s*([0-9.]+)", eval_text)
        if matches:
            return round(float(matches[-1]))

        # Backwards compatibility: legacy <score> tag
        legacy_tag_match = re.findall(r"<score>([0-9.]+)</score>", eval_text)
        if legacy_tag_match:
            return round(float(legacy_tag_match[-1]))

        # Older Tengu style: [点数]\n7点
        score_text_match = re.search(r"\[点数\]\\n[0-9.]+点", eval_text)
        if score_text_match:
            score_match_fallback = re.search(r"[0-9.]+", score_text_match.group())
            if score_match_fallback:
                return round(float(score_match_fallback.group()))

        raise ValueError("Could not find score pattern")

    except (ValueError, AttributeError):
        _log_no_score("TENGU", {}, eval_text, note="parse_error")
        return None

def make_tengu_conversation(data: dict) -> list:
    return [
        {
            "role": "user",
            "content": get_tengu_prompt(tengu_example_question_answer)
        },{
            "role": "assistant",
            "content": tengu_example_evaluation
        },{
            "role": "user",
            "content": get_tengu_prompt(data)
        }
    ]

def tengu_bench_evaluator(data:dict, model_name:str) -> dict:
    messages = make_tengu_conversation(data)
    evaluation = get_model_response(messages, model_name)
    score = get_tengu_eval_score(evaluation)
    if score is None:
        # Guard: if the judge forgot to emit a FINAL SCORE, ask it to
        # summarize its own analysis into a numeric score between 0 and 10.
        score = _fallback_score_from_analysis(
            evaluation,
            data,
            model_name,
            "between 0 and 10 (inclusive)",
        )

    return {
        "score": score,
        # Keep the full judge analysis so downstream tools can surface it.
        "judge_output": evaluation,
    }

######### ELYZA ##########

def get_elyza_prompt(row: dict):
    question = row['Question']
    answer = row['output']
    criteria = row['eval_aspect']
    model_answer = row['ModelAnswer']
    return f"""あなたは採点者です。

問題, 正解例, 採点基準, 回答 が与えられます。

採点基準と正解例を参考にして、回答を1,2,3,4,5の5段階で採点してください。
説明の後、最後の行に「FINAL SCORE: x」（x は最終スコアの数値）だけを出力してください。それ以外の行には「FINAL SCORE:」を書かないでください。

# 問題
{question}

# 正解例
{answer}

# 採点基準
基本的な採点基準
- 1点: 誤っている、 指示に従えていない
- 2点: 誤っているが、方向性は合っている
- 3点: 部分的に誤っている、 部分的に合っている
- 4点: 合っている
- 5点: 役に立つ

基本的な減点項目
- 不自然な日本語: -1点
- 部分的に事実と異なる内容を述べている: -1点
- 「倫理的に答えられません」のように過度に安全性を気にしてしまっている: 2点にする
- 日本語で回答していない: 1点にする
- 支離滅裂・意味不明な出力や反復/ループがある: 1点にする

問題固有の採点基準
{criteria}

# 回答
{model_answer}

FINAL SCORE: #
"""

def elyza_evaluator(data: dict, model_name:str) -> dict:
    prompt = get_elyza_prompt(data)
    messages = [{"role": "user", "content": prompt}]
    evaluation = get_model_response(messages, model_name)
    score: int | None
    try:
        # Prefer FINAL SCORE pattern on the last occurrence
        matches = re.findall(r"FINAL SCORE:\s*([0-9.]+)", evaluation)
        if matches:
            score = round(float(matches[-1]))
        else:
            # Backwards compatibility: legacy <score> tag
            legacy_tag_match = re.findall(r"<score>([0-9.]+)</score>", evaluation)
            if legacy_tag_match:
                score = round(float(legacy_tag_match[-1]))
            else:
                # Fallback: treat the whole output as a bare number
                score = round(float(evaluation.strip()))
    except (ValueError, AttributeError):
        _log_no_score("ELYZA", data, evaluation, note="parse_error")
        # Guard: if the judge forgot FINAL SCORE entirely, ask it to
        # reduce its own analysis to a numeric score between 1 and 5.
        score = _fallback_score_from_analysis(
            evaluation,
            data,
            model_name,
            "between 1 and 5 (inclusive)",
        )

    return {
        "score": score,
        "judge_output": evaluation,
    }

######### MT-Bench ##########

def get_mt_prompt(row: dict):
    question = row['Question']
    model_answer = row['ModelAnswer']
    return f"""[指示]
公平な判断者として行動し、以下に表示されるユーザーの質問に対するAIアシスタントの応答の品質を評価してください。あなたの評価は、応答の有用性、関連性、正確性、深さ、創造性、詳細度などの要素を考慮すべきです。AIアシスタントの返答は正しく自然な日本語であるべきで、そうでない場合、あるいは不適切な言語表現を含む場合は大きく減点してください。さらに、支離滅裂・意味不明な出力や反復/ループがある場合も大きく減点してください。評価は短い説明から始めてください。できるだけ客観的であること。説明を提供した後、1から10までのスケールで最終スコアを決定し、最後の行に「FINAL SCORE: x」（x は最終スコアの数値）だけを出力してください。それ以外の行には「FINAL SCORE:」を書かないでください。

[質問]
{question}

[アシスタントの回答の開始]
{model_answer}

[アシスタントの回答の終了]

IMPORTANT REMINDER: After judging, remember to output FINAL SCORE: # on it's own line.
"""

def mt_evaluator(data: dict, model_name:str) -> dict:
    prompt = get_mt_prompt(data)
    messages = [{"role": "user", "content": prompt}]
    evaluation = get_model_response(messages, model_name)
    score: int | None = None
    try:
        # Prefer FINAL SCORE pattern, using last occurrence
        matches = re.findall(r"FINAL SCORE:\s*([0-9.]+)", evaluation)
        if matches:
            score = round(float(matches[-1]))
        else:
            # Backwards compatibility: legacy <score> tag
            legacy_tag_match = re.findall(r"<score>([0-9.]+)</score>", evaluation)
            if legacy_tag_match:
                score = round(float(legacy_tag_match[-1]))
            else:
                # Backwards compatibility: legacy [[評価]] style
                score_text_match = re.search(r"評価：\[\[[0-9.]+\]\]", evaluation)
                if score_text_match:
                    score_match = re.search(r"[0-9.]+", score_text_match.group())
                    if score_match:
                        score = round(float(score_match.group()))
    except (ValueError, AttributeError):
        _log_no_score("MT-BENCH", data, evaluation, note="parse_error")
        # Guard: ask the judge to convert its own analysis to a score
        # between 1 and 10 if it forgot to include FINAL SCORE.
        score = _fallback_score_from_analysis(
            evaluation,
            data,
            model_name,
            "between 1 and 10 (inclusive)",
        )

    return {
        "score": score,
        "judge_output": evaluation,
    }

######### Rakuda Benchmark ##########

def get_rakuda_prompt(row: dict):
    question = row['Question']
    model_answer = row['ModelAnswer']
    return f"""[指示]
以下に表示されたユーザーの質問に対するAIアシスタントのパフォーマンスについて、あなたのフィードバックをお願いします。回答の有用性、関連性、正確性、詳細度、日本語能力を評価してください。まず、アシスタントの有用性、関連性、正確性、詳細度、日本語能力の評価を提供してください。評価の包括的な説明も提供してください。ユーザーは日本語しか話さないので日本語で書かれていない回答には低評価をつけてください。日本語であっても、支離滅裂・意味不明な出力や反復/ループがある場合は大きく減点してください。できるだけ客観的であること。説明を提供した後、1から10までのスケールで最終スコアを決定し、最後の行に「FINAL SCORE: x」（x は最終スコアの数値）だけを出力してください。それ以外の行には「FINAL SCORE:」を書かないでください。


[質問]
{question}

[アシスタントの回答の開始]
{model_answer}

[アシスタントの回答の終了]

FINAL SCORE: #

IMPORTANT REMINDER: After judging, remember to output FINAL SCORE: # on it's own line.
"""

def rakuda_evaluator(data: dict, model_name:str) -> dict:
    prompt = get_rakuda_prompt(data)
    messages = [{"role": "user", "content": prompt}]
    evaluation = get_model_response(messages, model_name)
    score: int | None
    try:
        # Prefer FINAL SCORE pattern, using last occurrence
        matches = re.findall(r"FINAL SCORE:\s*([0-9.]+)", evaluation)
        if matches:
            score = round(float(matches[-1]))
        else:
            # Backwards compatibility: legacy <score> tag
            legacy_tag_match = re.findall(r"<score>([0-9.]+)</score>", evaluation)
            if legacy_tag_match:
                score = round(float(legacy_tag_match[-1]))
            else:
                # Backwards compatibility: legacy [[5]] style
                score_text_match = re.search(r"評価：(\[\[|\[|【)[0-9.]+(\]\]|\]|】)", evaluation)
                if score_text_match:
                    score_match = re.search(r"[0-9.]+", score_text_match.group())
                    if score_match:
                        score = round(float(score_match.group()))
                else:
                    # Fallback: look for X/10 style ratings and average them
                    slash_scores = [float(s) for s in re.findall(r"([0-9.]+)\s*/\s*10", evaluation)]
                    if slash_scores:
                        score = round(sum(slash_scores) / len(slash_scores))
                    else:
                        raise ValueError("Could not parse score")
    except (ValueError, AttributeError):
        _log_no_score("RAKUDA", data, evaluation, note="parse_error")
        # Guard: as a last resort, ask the judge once more to emit a
        # single FINAL SCORE between 1 and 10 based on its analysis.
        score = _fallback_score_from_analysis(
            evaluation,
            data,
            model_name,
            "between 1 and 10 (inclusive)",
        )

    return {
        "score": score,
        "judge_output": evaluation,
    }

#### ALL EVAL DATASETS ####

EVAL_MODEL_CONFIGS = {
    "lightblue/tengu_bench": {
        "question_column": "Question",
        "evaluator_function": tengu_bench_evaluator,
        "split_name": "test"
    },
    "elyza/ELYZA-tasks-100": {
        "question_column": "input",
        "evaluator_function": elyza_evaluator,
        "split_name": "test"
    },
    "shisa-ai/ja-mt-bench-1shot": {
        "question_column": "Question",
        "evaluator_function": mt_evaluator,
        "split_name": "train"
    },
    "yuzuai/rakuda-questions": {
        "question_column": "text",
        "evaluator_function": rakuda_evaluator,
        "split_name": "train"
    },
}

import os

get_ans_path = lambda dataset_name, model_name: os.path.join(".", "data", "model_answers", dataset_name.replace("/", "__"), model_name.replace("/", "__") + ".json")


# -------- Dataset-level question exclusions --------
#
# Some evaluation datasets contain questions that we want to exclude from
# answer generation, judging, and aggregate scoring. We track these by a
# 1-based question id aligned with the dataset row order.

EXCLUDED_QUESTION_IDS_BY_DATASET: dict[str, set[int]] = {
    # Tengu-Bench: exclude safety-sensitive questions 61–65 whose
    # rubric is underspecified and highly judge-dependent.
    "lightblue/tengu_bench": {61, 62, 63, 64, 65},
}


def ensure_id_column(dataset, id_column: str = "id"):
    """
    Ensure the given dataset has an integer id column.

    - If the column already exists, the dataset is returned unchanged.
    - Otherwise, a 1-based sequential id is added in the current row order.
    """
    # HuggingFace Datasets expose `column_names`; for other dataset-like
    # objects we fall back to a no-op.
    column_names = getattr(dataset, "column_names", None)
    if column_names is None or id_column in column_names:
        return dataset

    return dataset.map(lambda _row, idx: {id_column: idx + 1}, with_indices=True)


def filter_excluded_questions_by_id(dataset_name: str, dataset, id_column: str = "id"):
    """
    Filter out rows whose question id is in the dataset's exclude list.

    This is used by:
    - answer generation (to avoid creating answers for excluded questions)
    - judging (to avoid submitting excluded questions to the judge)
    Downstream analysis tools can also use the same id lists when
    aggregating scores.
    """
    excluded_ids = EXCLUDED_QUESTION_IDS_BY_DATASET.get(dataset_name)
    if not excluded_ids:
        return dataset

    dataset = ensure_id_column(dataset, id_column=id_column)
    excluded_set = {int(i) for i in excluded_ids}

    def _keep(row: dict) -> bool:
        try:
            qid = int(row[id_column])
        except Exception:
            # If the id is missing or non-numeric, keep the row rather than
            # accidentally dropping data.
            return True
        return qid not in excluded_set

    # HuggingFace Datasets support .filter; for other dataset-like objects
    # this is expected to be a no-op or raise, in which case callers should
    # not use this helper.
    return dataset.filter(_keep)
