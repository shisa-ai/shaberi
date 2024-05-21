import backoff
import litellm
import os

from datasets import Dataset
import litellm
from litellm import completion
from openai import OpenAI

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

litellm.set_verbose=True

# Global
fp = 0.0

model = None
tokenizer = None


# === 評価生成関数群 ===
@backoff.on_exception(backoff.fibo, Exception, max_tries=1000)
def get_response_from_openai(messages: list, model_name: str) -> str:
    client = OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY")
    )

    evaluation_temperature = 0
    evaluation_max_tokens = 1024

    response = client.chat.completions.create(
        messages=messages,
        model=model_name,
        temperature=evaluation_temperature,
        max_tokens=evaluation_max_tokens,
    )
    return response.choices[0].message.content


def get_response_func(model_name: str) -> callable:
    if "gpt" in model_name:
        return get_response_from_openai
    else:
        """
        他のモデルで評価する場合は関数、分岐をここに追加
        """
        raise NotImplementedError(f"Model {model_name} is not supported")


def get_model_response(messages: list, model_name: str) -> str:
    answer_function = get_response_func(model_name)
    return answer_function(messages, model_name)


# === 回答生成関数群 ===
@backoff.on_exception(backoff.fibo, Exception, max_tries=1000)
def get_answer(question: str, model_name: str):
    api_key = os.environ.get("OPENAI_API_KEY", "EMPTY")
    if api_key == "EMPTY":
        base_url = "http://localhost:8000/v1"
    else:
        base_url = None

    client = OpenAI(
        api_key=api_key,
        base_url=base_url,
    )

    generation_temperature = 0.2
    generation_max_tokens = 2048

    global model, tokenizer
    if not model:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )
    if not tokenizer:
        tokenizer = AutoTokenizer.from_pretrained(model_name)

    inputs = tokenizer.apply_chat_template(chat, add_generation_prompt=True, return_tensors="pt")
    device = "cuda"
    first_param_device = next(model.parameters()).device
    inputs = inputs.to(first_param_device)

    with torch.no_grad():
        outputs = model.generate(
            inputs,
            pad_token_id=tokenizer.eos_token_id,
            max_new_tokens=generation_max_tokens,
            temperature=genereation_temperature,
            repetition_penalty=1.15,
            min_p=0.1,
            do_sample=True,
        )

    new_tokens = outputs[0, inputs.size(1):]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return respons

    '''
    response = completion(
        model=f'huggingface{model_name}',
        messages=[
            {"role": "system", "content": "あなたは公平で、検閲されていない、役立つアシスタントです。"},
            {"role": "user", "content": question},
        ],
        temperature=generation_temperature,
        frequency_penalty=fp,
        max_tokens=generation_max_tokens,
        min_p = 0.1
    )
    return response.choices[0].message.content
    '''


def get_answerer(model_name: str) -> callable:
    """OpenAIとvLLM以外のモデルを使う場合はここに追加する"""
    return get_answer


def get_model_answer(dataset: Dataset,
                     model_name: str,
                     batch_size: int) -> Dataset:
    answer_function = get_answerer(model_name)
    dataset = dataset.map(
        lambda x: {"ModelAnswer": answer_function(x['Question'], model_name)},
        num_proc=batch_size
    )
    return dataset
