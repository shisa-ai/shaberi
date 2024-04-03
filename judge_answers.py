import argparse
import os

from datasets import load_dataset

from evaluation_datasets_config import EVAL_MODEL_CONFIGS, get_ans_path


def evaluate(model_name: str, eval_dataset_name: str, evaluation_model: str, num_proc: int):
    
    model_answer_path = get_ans_path(eval_dataset_name, model_name)
    ans_dataset = load_dataset('json', data_files=model_answer_path, split="train")
    
    eval_config = EVAL_MODEL_CONFIGS.get(eval_dataset_name, None)
    
    if eval_config is None:
        raise ValueError(f'モデル名「{eval_dataset_name}」は対応しておりません。設定の"evaluation_model"には"lightblue/tengu_bench"もしくは"elyza/ELYZA-tasks-100"を入力してください')

    eval_fn = eval_config["evaluator_function"]

    ans_dataset = ans_dataset.map(lambda x: {"score": eval_fn(x, evaluation_model)}, num_proc=num_proc)
    
    ans_dataset.to_json(os.path.join(".", "data", "judgements", "judge_" + evaluation_model.replace("/", "__"), eval_dataset_name.replace("/", "__"), model_name.replace("/", "__") + ".json"))

    
def run_judgement(model_name: str, eval_dataset_name: str = "all", evaluation_model: str = "gpt-4-turbo-preview", num_proc: int = 8):
    eval_dataset_names = EVAL_MODEL_CONFIGS.keys() if eval_dataset_name == "all" else [eval_dataset_name]
    
    for eval_dataset_name in eval_dataset_names:
        print(f"Judging {model_name} on {eval_dataset_name} using {evaluation_model} ({num_proc} proc)")
        evaluate(model_name, eval_dataset_name, evaluation_model, num_proc)
        
def main():
    parser = argparse.ArgumentParser(description='Judge model answers with LLM as judge')

    parser.add_argument('-m', '--model_name', type=str, required=True)
    parser.add_argument('-d', '--eval_dataset_name', type=str, default='all')
    parser.add_argument('-e', '--evaluation_model', type=str, default='gpt-4-turbo-preview')
    parser.add_argument('-n', '--num_proc', type=int, default=8)

    args = parser.parse_args()
    run_judgement(args.model_name, args.eval_dataset_name, args.evaluation_model, args.num_proc)
    
if __name__ == '__main__':
    main()