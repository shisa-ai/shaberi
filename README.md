# japanese_llm_eval
A repo for evaluating Japanese LLMs　・　日本語LLMを評価するレポ

## 実行方法
### 実行方法1
`run_eval.sh`ファイルを編集し実行する。

以下の実行コードを実行してください。（shはbashの場合があります）
```
sh test.sh
```
### 実行方法2
直接関数を実行する。

### 実行用コードの説明
#### 1. 評価用データセットごとのモデルの回答生成用関数
```
OPENAI_API_KEY=[自分のOpenAIのAPIキー] python generate_answer.py \ 
    --model_name [評価したいLLMのモデル名] \
    --eval_dataset_name [評価用データセット名] \
    --num_proc [並列処理数]
```
`--model_name`（required）：評価したいLLMのモデル名。

`--eval_dataset_name`（`default="all"`）：評価用データセット名。設定しない場合対応している全ての評価用データセットについて実行します。

`--num_proc`（`default=8`）：並列処理数。実行環境に合わせて設定してください。設定しない場合の並列処理数は 8 です。

#### 2. モデルの回答の評価用関数
```
OPENAI_API_KEY=[自分のOpenAIのAPIキー] python judge_answers.py \ 
    --model_name [評価したいLLMのモデル名] \
    --eval_dataset_name [評価用データセット名] \
    --evaluation_model [評価用のLLMモデル名] \
    --num_proc [並列処理数]
```
`--model_name`（required）：評価したいLLMのモデル名。

`--eval_dataset_name`（`default="all"`）：評価用データセット名。設定しない場合対応している全ての評価用データセットについて実行します。

`--evaluation_model`（`default="gpt-4-turbo-preview"`）：評価用のLLMモデル名。設定しない場合`gpt-4-turbo-preview`を用いて評価します。

`--num_proc`（`default=8`）：並列処理数。実行環境に合わせて設定してください。設定しない場合の並列処理数は 8 です。

## 評価することができるLLMのモデル
1. `gpt-3.5-turbo-0125`
2. `gpt-4-turbo-preview`

## 評価に利用できるモデル
1. `gpt-4-turbo-preview`

## 評価用データセット名
1. `lightblue/tengu_bench`
2. `elyza/ELYZA-tasks-100`
3. `lightblue/japanes-mt-bench-oneshot`
4. `yuzuai/rakuda-questions`