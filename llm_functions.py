from openai import OpenAI
from datasets import Dataset
import os

def get_response_from_openai(messages:list, model_name:str) -> str:
    client = OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY")
    )
    response = client.chat.completions.create(
        messages = messages,
        model = model_name,
        temperature = 0,
    )
    return response.choices[0].message.content

def get_answer_from_openai(question:str, model_name:str) -> str:
    return get_response_from_openai([{"role": "user", "content": question}], model_name)

def get_answer_from_vllm(question:str, model_name:str) -> str:
    api_key = "EMPTY"
    api_base = "http://localhost:8000/v1"
    client = OpenAI(
        api_key=api_key,
        base_url=api_base,
    )
    completion = client.completions.create(model=model_name,
                                        prompt=question)
    return completion.choices[0].text

def get_answerer(model_name: str):
    if "gpt" in model_name:
        return get_answer_from_openai
    else:
        return get_answer_from_vllm

def get_model_answer(dataset: Dataset, model_name:str, batch_size: int) -> Dataset:
    answer_function = get_answerer(model_name)
    dataset = dataset.map(lambda x: {"ModelAnswer": answer_function(x['Question'], model_name)}, num_proc=batch_size)
    return dataset