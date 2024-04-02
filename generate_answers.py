from tqdm import tqdm
from openai import OpenAI
from datasets import load_dataset
from datasets import Dataset
import pandas as pd
import inspect
import os
from evaluation_datasets_config import EVAL_MODEL_CONFIGS, get_ans_path
from llm_functions import get_model_answer 
tqdm.pandas()

def load_model_dataset(evaluation_dataset_name: str) -> Dataset:
    eval_config = EVAL_MODEL_CONFIGS.get(evaluation_dataset_name, None)
    
    if eval_config is None:
        raise ValueError(f'モデル名「{evaluation_model}」は対応しておりません。設定の"evaluation_model"には"lightblue/tengu_bench"もしくは"elyza/ELYZA-tasks-100"を入力してください')
        
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
          
def main(model_name: str, eval_dataset_name: str = "all", request_batch_size: int = 16):
    
    eval_dataset_names = list(EVAL_MODEL_CONFIGS.keys()) if eval_dataset_name == "all" else [eval_dataset_name]
    
    for dataset_name in eval_dataset_names:
        # 1. テストデータセットの読み込み
        dataset = load_model_dataset(dataset_name)
        # 2. モデルの回答の取得
        dataset = get_model_answer(dataset, model_name, request_batch_size)
        model_answer_path = get_ans_path(dataset_name, model_name)
        dataset.to_json(model_answer_path)

main("gpt-3.5-turbo-0125", "all")