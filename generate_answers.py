import argparse

from datasets import Dataset, load_dataset

from evaluation_datasets_config import EVAL_MODEL_CONFIGS, get_ans_path
from llm_functions import get_model_answer


def load_model_dataset(evaluation_dataset_name: str) -> Dataset:
    eval_config = EVAL_MODEL_CONFIGS.get(evaluation_dataset_name, None)

    if eval_config is None:
        raise ValueError(f'モデル名「{evaluation_dataset_name}」は対応しておりません。引数の"--eval_dataset_name"は{list(EVAL_MODEL_CONFIGS.keys())}から選択してください。')

    # Load dataset
    dataset = load_dataset(evaluation_dataset_name)

    # Get specified split
    split_name = eval_config.get("split_name", "test")
    split_dataset = dataset[split_name]

    # Map question column to standardised column name
    q_col = eval_config.get("question_column")
    if q_col != "Question":
        split_dataset = split_dataset.rename_column(q_col, "Question")
    return split_dataset
          
def run_generate(model_name: str, eval_dataset_name: str = "all", num_proc: int = 16):
    
    eval_dataset_names = list(EVAL_MODEL_CONFIGS.keys()) if eval_dataset_name == "all" else [eval_dataset_name]
    
    for dataset_name in eval_dataset_names:
        # 1. テストデータセットの読み込み
        dataset = load_model_dataset(dataset_name)
        # 2. モデルの回答の取得
        dataset = get_model_answer(dataset, model_name, num_proc)
        model_answer_path = get_ans_path(dataset_name, model_name)
        dataset.to_json(model_answer_path)

def main():
    parser = argparse.ArgumentParser(description='Judge model answers with LLM as judge')

    parser.add_argument('-m', '--model_name', type=str, required=True)
    parser.add_argument('-d', '--eval_dataset_name', type=str, default='all')
    parser.add_argument('-n', '--num_proc', type=int, default=8)

    args = parser.parse_args()
    run_generate(args.model_name, args.eval_dataset_name, args.num_proc)
    
if __name__ == '__main__':
    main()