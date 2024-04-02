from datasets import load_dataset
from evaluation_datasets_config import EVAL_MODEL_CONFIGS, get_ans_path
import os

def evaluate(model_name: str, eval_dataset_name: str, evaluation_model: str, num_proc: int):
    
    model_answer_path = get_ans_path(eval_dataset_name, model_name)
    ans_dataset = load_dataset('json', data_files=model_answer_path, split="train")
    
    eval_config = EVAL_MODEL_CONFIGS.get(eval_dataset_name, None)
    
    if eval_config is None:
        raise ValueError(f'モデル名「{eval_dataset_name}」は対応しておりません。設定の"evaluation_model"には"lightblue/tengu_bench"もしくは"elyza/ELYZA-tasks-100"を入力してください')

    eval_fn = eval_config["evaluator_function"]

    ans_dataset = ans_dataset.map(lambda x: {"score": eval_fn(x, evaluation_model)}, num_proc=num_proc)
    
    ans_dataset.to_json(os.path.join(".", "data", "judgements", "judge_" + evaluation_model.replace("/", "__"), eval_dataset_name.replace("/", "__"), model_name.replace("/", "__") + ".json"))

    
def main(model_name: str, eval_dataset_name: str = "all", evaluation_model: str = "gpt-4-turbo-preview", num_proc: int = 8):
    eval_dataset_names = EVAL_MODEL_CONFIGS.keys() if eval_dataset_name == "all" else [eval_dataset_name]
    
    for eval_dataset_name in eval_dataset_names:
        print(f"Judging {model_name} on {eval_dataset_name} using {evaluation_model} ({num_proc} proc)")
        evaluate(model_name, eval_dataset_name, evaluation_model, num_proc)
        
main("gpt-3.5-turbo-0125", num_proc=12)