# Shaberi v2.1: A Suite of Japanese Chat Benchmarks

A repo for evaluating Japanese LLMs　・　日本語LLMを評価するレポ

This is a heavily modified version of [LightBlue's Shaberi](https://github.com/lightblue-tech/japanese_llm_eval) evaluation suite. It's due for an overhaul soon, hwever it *does* support quite a few addditional features:

- Full support for OpenAI API Compatible (vLLM/SGLang etc) models, as well as OpenAI and Gemini models (including custom parameters to support GPT5+, Gemini safety and reasoning, etc)
  - `backoff` used for handling retries and rate limits
- The ability to parse out reasoning/think tags
- Saving of judgement reasoning
- Heuristics for calculating a "JA %" to help detect non-JA output
- Saving of XLSX/CSV spreadsheets for output
- A `shaberi-viewer.py` TUI to allow easy inspection of outputs
- IMPORTANT: 2025-11: Modified prompts to try to more harshly penalize wrong language or corrupted output
  - There is a new `reprocess-forgotten-scores.py` to help when the LLM Judge forgets to output the final score.

Scoring should not be compared to other versions. Also note that different judges grade very differently. Any comparisons should be made with the same prompt and LLM judge.

We've found that the commonly used GPT-4/GPT-4 Turbo/GPT-4o judges are not adequate in 2025. In the past, Gemini judges have been too lenient, although the Gemini 2.5/3 Pro models are now (2025-11) the best Japanese models. For current judging we (Shisa.AI) are using GPT-5.1 which is stricter and more discerning on judging output.


## How to run
```
# Get code
git clone https://github.com/shisa-ai/shaberi
cd shaberi

# Create Environment, Install requirements
mamba create -n shaberi python=3.12
mamba activate shaberi
pip install -r requirements.txt

# For AMD or other hardware you may need to install torch manually
# https://pytorch.org/get-started/locally/
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.4

# In one terminal, run vLLM OpenAI API, eg: 
python -m vllm.entrypoints.openai.api_server --model shisa-ai/shisa-v1-llama3-70b -tp 8
# or llama.cpp OpenAI API, eg:
./server -ngl 99 -c 8192 -m shisa-v1-llama3-70b.Q4_K_M.gguf --chat-template llama3 --host 0.0.0.0 --port 8000 -a shisa-v1-llama3-70b.q4_k_m

# In a separate terminal, generate answers:
mamba activate shaberi
# Match model name to what vLLM is serving
# We run frequency_penalty=0.5 for all our runs, probably generally the best
python generate_answers.py --model_name 'shisa-ai/shisa-v1-llama3-8b' -fp 0.5

# Then run the judge (assumes your OPENAI_API_KEY is in the env already):
python judge_answers.py -m shisa-ai/shisa-v1-llama3-8b

# Make sure you have new answers and judgements
git status

# To generate updated results
python results_vizualization.py
cat output.csv
```

---
```
### OLD SHABERI README... 
```


## 実行方法
### 1. 評価用データセットごとのモデルの回答生成用関数
```
OPENAI_API_KEY=[自分のOpenAIのAPIキー] python generate_answer.py \ 
    --model_name [評価したいLLMのモデル名] \
    --eval_dataset_name [評価用データセット名] \
    --num_proc [並列処理数]
```
`--model_name`（required）：評価したいLLMのモデル名。

`--eval_dataset_name`（`default="all"`）：評価用データセット名。設定しない場合対応している全ての評価用データセットについて実行します。

`--num_proc`（`default=8`）：並列処理数。実行環境に合わせて設定してください。設定しない場合の並列処理数は 8 です。

### 2. モデルの回答の評価用関数
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
OpenAI社のモデルおよび、vLLM等のopenaiモジュール形式のサーバーを立てることができるツールが対応しているモデル
> vLLMでopenai形式のサーバーを立てる方法については公式ドキュメントをご覧ください→https://docs.vllm.ai/en/latest/getting_started/quickstart.html

## 評価に利用できるモデル
1. `gpt-4-turbo-preview`

## 評価用データセット名
1. `lightblue/tengu_bench`: Tengu-Bench
2. `elyza/ELYZA-tasks-100`: Elyza-tasks-100
3. `lightblue/japanes-mt-bench-oneshot`: MT-Bench
4. `yuzuai/rakuda-questions`: Rakuda

## 結果
|                                            |   ELYZA-tasks-100 |   Rakuda |   Tengu-Bench |   MT-Bench |   mean |
|:-------------------------------------------|------------------:|---------:|--------------:|-----------:|-------:|
| gpt-4-turbo-2024-04-09                     |              8.78 |     9.18 |          **8.31** |       **8.74** |   **8.75** |
| gpt-4-turbo-preview                        |              **8.94** |     **9.28** |          7.84 |       8.61 |   8.67 |
| CohereForAI__c4ai-command-r-plus           |              7.50 |     9.05 |          6.79 |       7.42 |   7.69 |
| Qwen__Qwen1.5-72B-Chat                     |              7.60 |     7.85 |          6.81 |       7.16 |   7.36 |
| gpt-3.5-turbo-0125                         |              7.24 |     7.64 |          6.82 |       6.97 |   7.17 |
| CohereForAI__c4ai-command-r-v01            |              6.08 |     8.62 |          6.67 |       6.94 |   7.08 |
| Qwen__Qwen1.5-32B-Chat                     |              7.09 |     7.51 |          6.36 |       6.90 |   6.97 |
| karakuri-ai__karakuri-lm-70b-chat-v0.1     |              6.86 |     7.85 |          6.23 |       6.43 |   6.84 |
| lightblue__ao-karasu-72B                   |              7.19 |     7.25 |          6.27 |       6.54 |   6.81 |
| Qwen__Qwen1.5-14B-Chat                     |              6.70 |     6.54 |          6.20 |       6.54 |   6.50 |
| xverse__XVERSE-13B-Chat                    |              6.34 |     6.65 |          4.88 |       5.34 |   5.80 |
| Rakuten__RakutenAI-7B-chat                 |              5.92 |     6.58 |          5.24 |       4.60 |   5.58 |
| Nexusflow__Starling-LM-7B-beta             |              5.74 |     4.42 |          5.41 |       5.61 |   5.30 |
| Qwen__Qwen1.5-7B-Chat                      |              5.54 |     4.82 |          5.29 |       5.41 |   5.27 |
| elyza__ELYZA-japanese-Llama-2-13b-instruct |              5.60 |     5.62 |          5.52 |       4.31 |   5.26 |
| lightblue__qarasu-14B-chat-plus-unleashed  |              5.58 |     5.46 |          5.01 |       4.74 |   5.20 |
| openchat__openchat-3.5-0106                |              5.82 |     3.77 |          5.49 |       5.04 |   5.03 |
| cyberagent__calm2-7b-chat                  |              4.90 |     5.75 |          4.81 |       3.58 |   4.76 |
| mistralai__Mistral-7B-Instruct-v0.2        |              5.78 |     3.80 |          4.53 |       4.65 |   4.69 |
| meta-llama__Llama-2-13b-chat-hf            |              5.64 |     2.27 |          4.67 |       5.71 |   4.58 |
| meta-llama__Llama-2-7b-chat-hf             |              4.78 |     2.08 |          3.92 |       5.45 |   4.06 |
| augmxnt__shisa-7b-v1                       |              3.72 |     2.23 |          3.41 |       2.23 |   2.89 |

<details><summary>カテゴリーごとの詳しい結果はこちらをご覧ください。</summary>

| model_name                                 |   ('MT-Bench', 'coding') |   ('MT-Bench', 'extraction') |   ('MT-Bench', 'humanities') |   ('MT-Bench', 'math') |   ('MT-Bench', 'reasoning') |   ('MT-Bench', 'roleplay') |   ('MT-Bench', 'stem') |   ('MT-Bench', 'writing') |   ('Tengu-Bench', 'Function calling') |   ('Tengu-Bench', 'アイデア生成') |   ('Tengu-Bench', 'コスト見積') |   ('Tengu-Bench', 'ダジャレ') |   ('Tengu-Bench', 'ビジネス') |   ('Tengu-Bench', 'フォーマット') |   ('Tengu-Bench', 'プロジェクト作成') |   ('Tengu-Bench', '会話要約') |   ('Tengu-Bench', '倫理的制御') |   ('Tengu-Bench', '建設') |   ('Tengu-Bench', '抽出') |   ('Tengu-Bench', '政治') |   ('Tengu-Bench', '敬語') |   ('Tengu-Bench', '数学') |   ('Tengu-Bench', '日本') |   ('Tengu-Bench', '架空の質問') |   ('Tengu-Bench', '法律判断') |   ('Tengu-Bench', '翻訳') |   ('Tengu-Bench', '表の読み取り') |   ('Tengu-Bench', '論理パズル') |   ('Tengu-Bench', '長い文書のClosed QA（千トークン以上）') |   ('Tengu-Bench', '長い文書要約（千トークン以上）') |   ('Tengu-Bench', '雑談') |
|:-------------------------------------------|-------------------------:|-----------------------------:|-----------------------------:|-----------------------:|----------------------------:|---------------------------:|-----------------------:|--------------------------:|--------------------------------------:|----------------------------------:|--------------------------------:|------------------------------:|------------------------------:|----------------------------------:|--------------------------------------:|------------------------------:|--------------------------------:|--------------------------:|--------------------------:|--------------------------:|--------------------------:|--------------------------:|--------------------------:|--------------------------------:|------------------------------:|--------------------------:|----------------------------------:|--------------------------------:|-----------------------------------------------------------:|----------------------------------------------------:|--------------------------:|
| CohereForAI__c4ai-command-r-plus           |                     6.10 |                         8.60 |                         8.60 |                   5.60 |                        5.70 |                       8.20 |                   7.80 |                      8.80 |                                  8.20 |                             10.00 |                            9.00 |                          3.60 |                          2.60 |                              9.00 |                                 10.00 |                         10.00 |                            2.80 |                      7.00 |                      9.40 |                      5.60 |                      9.20 |                      1.20 |                      4.30 |                            3.60 |                          7.60 |                      8.40 |                              5.40 |                            2.33 |                                                       8.20 |                                               10.00 |                      9.40 |
| CohereForAI__c4ai-command-r-v01            |                     6.90 |                         6.60 |                         8.40 |                   4.70 |                        5.20 |                       7.67 |                   7.70 |                      8.40 |                                  9.20 |                             10.00 |                            9.80 |                          4.60 |                          4.80 |                              9.00 |                                 10.00 |                         10.00 |                            3.40 |                      5.40 |                      8.80 |                      4.60 |                      9.20 |                      1.00 |                      4.50 |                            3.60 |                          6.60 |                      8.40 |                              1.80 |                            2.60 |                                                       8.60 |                                               10.00 |                      9.80 |
| Nexusflow__Starling-LM-7B-beta             |                     5.30 |                         6.30 |                         6.10 |                   5.50 |                        5.10 |                       5.50 |                   4.10 |                      7.00 |                                  6.60 |                              9.00 |                            5.40 |                          4.20 |                          3.00 |                              8.00 |                                  9.20 |                          9.20 |                            4.00 |                      4.80 |                      9.20 |                      1.20 |                      6.80 |                      1.40 |                      2.10 |                            6.00 |                          2.40 |                      7.60 |                              1.80 |                            1.80 |                                                       6.80 |                                               10.00 |                      7.20 |
| Qwen__Qwen1.5-14B-Chat                     |                     5.40 |                         7.40 |                         7.00 |                   5.40 |                        5.30 |                       7.20 |                   7.00 |                      7.60 |                                  6.40 |                             10.00 |                            8.80 |                          4.40 |                          4.60 |                              7.80 |                                  9.40 |                         10.00 |                            4.60 |                      5.20 |                      9.00 |                      2.80 |                      8.60 |                      1.40 |                      3.40 |                            5.20 |                          5.00 |                      4.80 |                              3.40 |                            3.40 |                                                       9.60 |                                               10.00 |                      7.60 |
| Qwen__Qwen1.5-32B-Chat                     |                     5.70 |                         7.40 |                         7.60 |                   6.20 |                        6.00 |                       7.20 |                   7.20 |                      7.90 |                                  8.20 |                             10.00 |                            8.80 |                          5.20 |                          2.60 |                              6.00 |                                 10.00 |                         10.00 |                            3.40 |                      4.40 |                      9.80 |                      2.80 |                      9.00 |                      3.80 |                      3.60 |                            6.00 |                          5.40 |                      7.80 |                              6.00 |                            3.00 |                                                       7.40 |                                                9.00 |                      7.60 |
| Qwen__Qwen1.5-72B-Chat                     |                     6.70 |                         6.90 |                         7.70 |                   6.80 |                        4.80 |                       8.00 |                   7.90 |                      8.50 |                                  5.60 |                             10.00 |                            9.60 |                          3.60 |                          2.60 |                              7.80 |                                 10.00 |                         10.00 |                            5.80 |                      7.40 |                      9.60 |                      4.20 |                      8.20 |                      4.00 |                      2.90 |                            7.60 |                          6.00 |                      7.80 |                              6.20 |                            3.40 |                                                       9.60 |                                               10.00 |                      8.60 |
| Qwen__Qwen1.5-7B-Chat                      |                     4.90 |                         6.30 |                         5.20 |                   4.70 |                        4.30 |                       5.80 |                   5.00 |                      7.10 |                                  5.00 |                              9.60 |                            7.60 |                          5.80 |                          2.40 |                              4.80 |                                  7.80 |                          9.60 |                            4.40 |                      3.40 |                      8.40 |                      2.00 |                      7.60 |                      0.80 |                      1.80 |                            5.20 |                          3.00 |                      6.60 |                              3.60 |                            1.20 |                                                       9.00 |                                                9.40 |                      6.20 |
| Rakuten__RakutenAI-7B-chat                 |                     4.70 |                         3.90 |                         6.80 |                   4.20 |                        3.30 |                       4.60 |                   4.20 |                      5.10 |                                  3.20 |                              9.80 |                            8.60 |                          3.20 |                          2.80 |                              4.60 |                                 10.00 |                          6.80 |                            7.60 |                      4.40 |                      7.40 |                      2.40 |                      5.20 |                      1.00 |                      4.20 |                            4.40 |                          4.80 |                      6.60 |                              3.40 |                            2.00 |                                                       5.40 |                                                8.00 |                      5.80 |
| augmxnt__shisa-7b-v1                       |                     3.50 |                         4.00 |                         1.70 |                   3.30 |                        1.90 |                       1.10 |                   1.00 |                      1.30 |                                  2.40 |                              6.00 |                            4.40 |                          1.00 |                          2.20 |                              2.20 |                                  7.80 |                          6.40 |                            2.40 |                      1.20 |                      6.40 |                      1.00 |                      3.80 |                      1.00 |                      1.70 |                            2.80 |                          1.60 |                      3.60 |                              1.20 |                            1.60 |                                                       5.80 |                                                9.80 |                      3.80 |
| cyberagent__calm2-7b-chat                  |                     2.30 |                         4.10 |                         6.10 |                   1.30 |                        2.40 |                       4.30 |                   4.40 |                      3.70 |                                  4.40 |                              9.00 |                            5.20 |                          3.60 |                          3.00 |                              3.40 |                                  9.20 |                          7.20 |                            2.80 |                      2.80 |                      7.60 |                      3.40 |                      3.60 |                      1.00 |                      3.40 |                            5.20 |                          5.80 |                      5.20 |                              1.00 |                            1.00 |                                                       6.40 |                                                9.60 |                      8.20 |
| elyza__ELYZA-japanese-Llama-2-13b-instruct |                     3.20 |                         5.20 |                         5.60 |                   2.90 |                        4.00 |                       4.70 |                   4.40 |                      4.50 |                                  9.60 |                              8.60 |                            6.60 |                          5.80 |                          3.80 |                              5.80 |                                 10.00 |                          6.80 |                            2.80 |                      3.80 |                      7.60 |                      5.00 |                      6.00 |                      0.80 |                      2.60 |                            4.40 |                          5.40 |                      5.60 |                              1.80 |                            1.80 |                                                       7.20 |                                                9.80 |                      8.20 |
| gpt-3.5-turbo-0125                         |                     7.00 |                         8.80 |                         7.30 |                   6.80 |                        4.20 |                       7.40 |                   7.00 |                      7.30 |                                  6.80 |                             10.00 |                            9.20 |                          4.00 |                          4.60 |                              9.60 |                                 10.00 |                          9.20 |                            3.40 |                      4.80 |                     10.00 |                      2.20 |                      7.80 |                      4.00 |                      3.90 |                            6.00 |                          6.80 |                      8.60 |                              7.80 |                            4.80 |                                                       8.00 |                                                9.80 |                      8.40 |
| gpt-4-turbo-2024-04-09                     |                     8.50 |                         9.10 |                         8.80 |                   9.50 |                        7.70 |                       8.60 |                   8.80 |                      8.90 |                                  8.40 |                             10.00 |                            9.60 |                          8.20 |                          6.20 |                             10.00 |                                 10.00 |                         10.00 |                           10.00 |                      6.40 |                      9.60 |                      6.00 |                      9.60 |                      7.60 |                      4.90 |                            9.20 |                          6.80 |                      9.00 |                              8.80 |                            5.80 |                                                       8.60 |                                               10.00 |                      9.80 |
| gpt-4-turbo-preview                        |                     8.10 |                         9.00 |                         8.90 |                   8.50 |                        8.00 |                       8.70 |                   8.60 |                      9.10 |                                 10.00 |                             10.00 |                            9.80 |                          4.00 |                          6.60 |                             10.00 |                                 10.00 |                         10.00 |                            8.20 |                      6.40 |                     10.00 |                      6.40 |                      9.60 |                      6.60 |                      4.20 |                            6.00 |                          6.00 |                      8.80 |                              8.40 |                            4.20 |                                                       9.00 |                                               10.00 |                      9.80 |
| karakuri-ai__karakuri-lm-70b-chat-v0.1     |                     5.90 |                         6.90 |                         8.30 |                   4.10 |                        5.09 |                       6.80 |                   6.30 |                      8.20 |                                  6.00 |                              8.20 |                            8.20 |                          4.00 |                          4.40 |                              6.80 |                                 10.00 |                          9.60 |                            2.80 |                      5.40 |                      8.80 |                      4.20 |                      7.80 |                      2.20 |                      4.20 |                            3.60 |                          8.00 |                      8.40 |                              2.60 |                            2.60 |                                                       8.60 |                                                9.20 |                      9.80 |
| lightblue__ao-karasu-72B                   |                     6.00 |                         7.30 |                         7.50 |                   5.00 |                        5.60 |                       6.50 |                   6.60 |                      7.70 |                                  7.80 |                             10.00 |                            8.40 |                          5.40 |                          3.00 |                              6.20 |                                  9.00 |                          8.80 |                            2.60 |                      5.60 |                     10.00 |                      4.60 |                      6.60 |                      5.20 |                      3.90 |                            5.20 |                          6.20 |                      7.80 |                              4.40 |                            3.00 |                                                       6.60 |                                               10.00 |                      6.20 |
| lightblue__qarasu-14B-chat-plus-unleashed  |                     3.90 |                         6.60 |                         5.60 |                   4.30 |                        4.40 |                       3.10 |                   5.30 |                      4.70 |                                  4.40 |                             10.00 |                            6.80 |                          4.20 |                          2.00 |                              4.40 |                                  7.40 |                          9.00 |                            2.80 |                      5.80 |                      8.60 |                      2.40 |                      5.40 |                      2.80 |                      1.30 |                            5.20 |                          6.60 |                      6.60 |                              3.00 |                            0.80 |                                                       7.40 |                                                7.00 |                      5.00 |
| meta-llama__Llama-2-13b-chat-hf            |                     3.60 |                         6.00 |                         8.70 |                   3.50 |                        2.90 |                       5.90 |                   7.90 |                      7.20 |                                  7.80 |                              9.40 |                            7.80 |                          2.40 |                          2.20 |                              4.00 |                                  8.40 |                          4.80 |                            9.60 |                      4.60 |                      2.20 |                      1.40 |                      6.60 |                      0.40 |                      1.80 |                            2.80 |                          4.80 |                      6.80 |                              2.80 |                            1.40 |                                                       4.80 |                                                9.20 |                      4.40 |
| meta-llama__Llama-2-7b-chat-hf             |                     4.20 |                         5.60 |                         8.10 |                   2.70 |                        3.20 |                       6.10 |                   6.40 |                      7.30 |                                  3.00 |                              9.40 |                            7.40 |                          1.60 |                          0.60 |                              3.80 |                                  8.80 |                          5.00 |                            6.00 |                      3.80 |                      2.80 |                      0.20 |                      5.40 |                      0.80 |                      1.90 |                            6.00 |                          3.40 |                      5.00 |                              2.40 |                            1.40 |                                                       2.60 |                                                5.60 |                      5.20 |
| mistralai__Mistral-7B-Instruct-v0.2        |                     4.90 |                         4.70 |                         5.50 |                   2.60 |                        3.70 |                       5.20 |                   4.20 |                      6.40 |                                  6.80 |                              7.80 |                            7.80 |                          3.40 |                          1.60 |                              4.80 |                                  8.80 |                          7.40 |                            2.20 |                      3.40 |                      9.40 |                      1.20 |                      5.40 |                      0.80 |                      1.00 |                            3.60 |                          2.60 |                      5.60 |                              0.80 |                            0.20 |                                                       9.00 |                                                9.20 |                      5.00 |
| openchat__openchat-3.5-0106                |                     5.40 |                         5.90 |                         5.20 |                   4.40 |                        4.70 |                       5.20 |                   4.60 |                      4.90 |                                  7.60 |                              9.00 |                            6.00 |                          2.00 |                          3.40 |                              7.20 |                                  9.40 |                         10.00 |                            2.80 |                      3.80 |                     10.00 |                      2.60 |                      5.00 |                      0.80 |                      2.20 |                            5.20 |                          4.60 |                      7.60 |                              3.20 |                            1.80 |                                                       8.80 |                                                9.20 |                      7.40 |
| xverse__XVERSE-13B-Chat                    |                     4.60 |                         6.70 |                         5.50 |                   4.00 |                        3.30 |                       6.10 |                   5.90 |                      6.60 |                                  2.00 |                             10.00 |                            6.60 |                          3.00 |                          2.40 |                              5.60 |                                  9.40 |                          7.20 |                            0.80 |                      4.20 |                      8.00 |                      4.40 |                      5.60 |                      0.80 |                      1.30 |                            4.40 |                          5.80 |                      7.20 |                              2.20 |                            1.80 |                                                       8.00 |                                                7.80 |                      7.40 |

</details>

## 判明しているエラー
--num_proc=8でmeta-llama/Llama-2-7b-chat-hfを用いてlightblue/tengu_benchの回答を生成する際にエラーが発生する

→ --num_proc=1ならエラーが出ない（原因不明：tengu-benchのmax-tokenは2600程度だったので1024に変えたら大丈夫なはずだがダメだった）
