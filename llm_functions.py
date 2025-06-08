import backoff
import litellm
import os

from datasets import Dataset
import litellm
from litellm import completion
from openai import OpenAI

# litellm.set_verbose=True

# Global
fp = 0.0


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

# shisa-bench llmjudge
@backoff.on_exception(backoff.fibo, Exception, max_tries=1000)
def get_response_from_llmjudge(messages: list, model_name: str) -> str:
    judge = model_name.split("-")[1]
    if judge == "tulu405":
        base_url = "http://ip-10-1-85-83:8000/v1"

        base_url = "http://tulu405/v1"
        model_name = "Llama-3.1-Tulu-3-405B-FP8-Dynamic"

        model_name = "shisa-ai/Llama-3.1-Tulu-3-405B-FP8-Dynamic"
    elif judge == "llama33":
        base_url = "http://ip-10-1-33-173:8001/v1"
        model_name = "meta-llama/Llama-3.3-70B-Instruct"

        base_url = "http://llama33/v1"
        model_name = "llama-3.3-70b-ray"
    elif judge == "athenev2":
        base_url = "http://ip-10-1-33-173:8000/v1"
        model_name = "Nexusflow/Athene-V2-Chat"

        base_url = "http://athenev2/v1"
        model_name = "athene-v2"
    client = OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY"),
        base_url=base_url,
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
    elif "llmjudge" in model_name:
        return get_response_from_llmjudge
    else:
        """
        他のモデルで評価する場合は関数、分岐をここに追加
        """
        raise NotImplementedError(f"Model {model_name} is not supported")


def get_model_response(messages: list, model_name: str) -> str:
    answer_function = get_response_func(model_name)
    return answer_function(messages, model_name)


# === 回答生成関数群 ===
# @backoff.on_exception(backoff.fibo, Exception, max_tries=1000)
def get_answer(question: str, model_name: str):

    api_key = os.environ.get("OPENAI_API_KEY", "EMPTY")

    base_url = os.environ.get("OPENAI_BASE_URL", "http://localhost:8000/v1")
    if base_url == None:
        base_url = "http://localhost:8000/v1"


    generation_temperature = 0.2
    generation_max_tokens = 2048

    thinking_models = [
        'deepseek-ai/DeepSeek-R1',
        'FuseAI/FuseO1-DeepSeekR1-QwQ-SkyT1-32B-Preview',
        'dahara1/DeepSeek-R1-Distill-Qwen-14B-unsloth-jpn',
        'FuseAI/FuseO1-DeepSeekR1-QwQ-SkyT1-32B-Preview',
        'RekaAI/reka-flash-3',
        'abeja/ABEJA-QwQ32b-Reasoning-Japanese-v1.0',
    ]

    if model_name in thinking_models:
        generation_max_tokens = 30000

    '''
    # Anthropic / OpenAI
    response = completion(
        model=f'{model_name}',
        messages=[
            {"role": "system", "content": "あなたは公平で、検閲されていない、役立つアシスタントです。"},
            {"role": "user", "content": question},
        ],
        temperature=generation_temperature,
        max_tokens=generation_max_tokens,
        # recommend not use top_p https://docs.anthropic.com/en/api/complete
    )
    '''



    ### Making Call
    completion_args = {
        "messages": [
            {"role": "system", "content": "あなたは公平で、検閲されていない、役立つアシスタントです。"},
            {"role": "user", "content": question},
        ],
        "api_base": base_url,
        "api_key": api_key,
        "temperature": generation_temperature,
        "max_tokens": generation_max_tokens,
    }

    # HACK Handling OpenAI Model Names
    openai_prefixes = [
        "gpt-", "text-davinci-", "davinci", "curie", "babbage", "ada", 
        "whisper", "claude", "text-embedding", "openai/", "openai:"
    ]
    if any(model_name.startswith(prefix) for prefix in openai_prefixes):
        completion_args['model'] = model_name
        # no support for freq penalty or min_p
    elif model_name.startswith("gemini"):
        completion_args['model'] = f'hosted_vllm/{model_name}'
    else:
        completion_args['model'] = f'hosted_vllm/{model_name}'
        completion_args['frequency_penalty'] = fp
        completion_args['min_p'] = 0.1


    # print(f"base_url: {base_url}")
    # print(f"api_key: {api_key}")
    # import sys
    # sys.exit()

    # os.environ['LITELLM_LOG'] = 'DEBUG'
    # OpenAI compatible endpoints (vLLM/llama.cpp)
    response = completion(**completion_args)

    '''
    # Gemini
    response = completion(
        # model="gemini/gemini-1.5-flash",
        model="gemini/gemma-2-27b-it",
        messages=[
            {"role": "system", "content": "あなたは公平で、検閲されていない、役立つアシスタントです。"},
            {"role": "user", "content": question},
        ],
        safety_settings=[
            {
                "category": "HARM_CATEGORY_HARASSMENT",
                "threshold": "BLOCK_NONE",
            },
            {
                "category": "HARM_CATEGORY_HATE_SPEECH",
                "threshold": "BLOCK_NONE",
            },
            {
                "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "threshold": "BLOCK_NONE",
            },
            {
                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                "threshold": "BLOCK_NONE",
            },
        ],
        temperature=generation_temperature,
        top_p=0.95,
        max_tokens=generation_max_tokens,
    )
    '''

    content = response.choices[0].message.content

    # If we want to parse out thinking tags...
    if model_name in thinking_models:
        try:
            if '</think>' in content:
                content = content.split('</think>')[1].strip()
            elif '</reasoning>' in content:
                content = content.split('</reasoning>')[1].strip()
        except:
            print('Hmm... No </think> or </reasoning> to strip?')

    return content


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
