import re

from llm_functions import get_model_response

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
7点"""

def get_tengu_prompt(data: dict) -> str:
    question = data['Question']
    example_answer = data['Answer']
    criteria = data['Criteria']
    model_answer = data['ModelAnswer']

    answer_bool = example_answer is not None
    
    example_answer = f"\n[正解例]\n{example_answer}\n" if answer_bool else ""
    answer_explanation = "'正解例'," if answer_bool else ""
    
    prompt = f"""[指示]
あなたは熟練した生成AIモデルの性能評価者です。評価項目に準拠して客観的に評価することで報酬を得ることができますが、評価項目と異なる評価をした場合報酬がもらえなくなってしまいます。
以下の'質問',{answer_explanation}'評価項目'に基づいて'評価するモデルの回答'を0~10点の整数値で評価してください。

[質問]
{question}
{example_answer}
[評価項目]
{criteria}

[評価するモデルの回答]
{model_answer}

# 以下の形式で回答してください。
[該当する評価項目とその簡潔な理由]

[計算式]

[点数]
"""
    return prompt

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

def get_tengu_eval_score(eval_text: str) -> int:
    try:
        score_text = re.search(r"\[点数\]\n\d{1,2}点", eval_text).group()
        score = re.search(r"\d{1,2}", score_text).group()
        return int(score)
    except (ValueError, AttributeError):
        print(f"Unable to parse Tengu score from {eval_text}")
        return None
        
# Takes a dict and outputs a score for each
def tengu_bench_evaluator(data:dict, model_name:str) -> int|None:
    messages = make_tengu_conversation(data)
    evaluation = get_model_response(messages, model_name)
    return get_tengu_eval_score(evaluation)

######### ELYZA ##########

def get_elyza_prompt(row: dict):
    question = row['Question']
    answer = row['output']
    criteria = row['eval_aspect']
    model_answer = row['ModelAnswer']
    return f"""あなたは採点者です。

問題, 正解例, 採点基準, 回答 が与えられます。

採点基準と正解例を参考にして、回答を1,2,3,4,5の5段階で採点し、数字のみを出力してください。

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

問題固有の採点基準
{criteria}

# 回答
{model_answer}
"""

def elyza_evaluator(data: dict, model_name:str) -> int|None:
    prompt = get_elyza_prompt(data)
    messages = [{"role": "user", "content": prompt}]
    evaluation = get_model_response(messages, model_name)
    try:
        gpt4score = int(evaluation)
    except ValueError:
        print(f"Int parse error.\n\nOutput was {evaluation}.\n\nInput was {data}.")
        gpt4score = None
    return gpt4score

######### MT-Bench ##########

def get_mt_prompt(row: dict):
    question = row['Question']
    model_answer = row['ModelAnswer']
    return f"""[指示]
公平な判断者として行動し、以下に表示されるユーザーの質問に対するAIアシスタントの応答の品質を評価してください。あなたの評価は、応答の有用性、関連性、正確性、深さ、創造性、詳細度などの要素を考慮すべきです。AIアシスタントの返答の言語は、ユーザーが使用している言語と一致しているべきで、そうでない場合は減点されるべきです。評価は短い説明から始めてください。できるだけ客観的であること。説明を提供した後、このフォーマットに厳密に従って1から10までのスケールで応答を評価する必要があります："[[評価]]"、例えば："評価：[[5]]"。

[質問]
{question}

[アシスタントの回答の開始]
{model_answer}

[アシスタントの回答の終了]"""

def mt_evaluator(data: dict, model_name:str) -> int|None:
    prompt = get_mt_prompt(data)
    messages = [{"role": "user", "content": prompt}]
    evaluation = get_model_response(messages, model_name)
    try:
        score_text = re.search(r"評価：\[\[\d{1,2}\]\]", evaluation).group()
        score = re.search(r"\d{1,2}", score_text).group()
        return int(score)
    except (ValueError, AttributeError):
        print(f"Int parse error.\n\nOutput was {evaluation}.\n\nInput was {data}.")
        gpt4score = None
    return gpt4score

######### Rakuda Benchmark ##########

def get_rakuda_prompt(row: dict):
    question = row['Question']
    model_answer = row['ModelAnswer']
    return f"""[指示]
以下に表示されたユーザーの質問に対するAIアシスタントのパフォーマンスについて、あなたのフィードバックをお願いします。回答の有用性、関連性、正確性、詳細度、日本語能力を評価してください。まず、アシスタントの有用性、関連性、正確性、詳細度、日本語能力の評価を提供してください。評価の包括的な説明も提供してください。ユーザーは日本語しか話さないので日本語で書かれていない回答には低評価をつけてください。できるだけ客観的であること。説明を提供した後、このフォーマットに厳密に従って1から10までのスケールで応答を評価する必要があります："[[評価]]"、例えば："評価：[[5]]"。

[質問]
{question}

[アシスタントの回答の開始]
{model_answer}

[アシスタントの回答の終了]"""

def rakuda_evaluator(data: dict, model_name:str) -> int|None:
    prompt = get_rakuda_prompt(data)
    messages = [{"role": "user", "content": prompt}]
    evaluation = get_model_response(messages, model_name)
    try:
        score_text = re.search(r"評価：\[\[\d{1,2}\]\]", evaluation).group()
        score = re.search(r"\d{1,2}", score_text).group()
        return int(score)
    except (ValueError, AttributeError):
        print(f"Int parse error.\n\nOutput was {evaluation}.\n\nInput was {data}.")
        gpt4score = None
    return gpt4score

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
