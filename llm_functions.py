import backoff
import os

from datasets import Dataset
from openai import OpenAI

# Optional native Gemini support
try:
    import google.genai as genai
    from google.genai import types as genai_types
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

# litellm.set_verbose=True

# Global
fp = 0.0


# Helper: detect non-retryable safety/invalid prompt refusals
def _is_openai_refusal_error(err: Exception) -> bool:
    try:
        s = str(err).lower()
    except Exception:
        return False
    # Key indicators seen in OpenAI 400 errors for safety refusals
    # Example: BadRequestError: Error code: 400 - { 'type': 'invalid_request_error', 'code': 'invalid_prompt' }
    if "invalid_prompt" in s:
        return True
    if "we've limited access to this content" in s:
        return True
    # Be conservative: only treat as refusal when clearly indicated, not all invalid_request_error
    return False


# === 評価生成関数群 ===
@backoff.on_exception(backoff.fibo, Exception, max_tries=1000)
def get_response_from_openai(messages: list, model_name: str) -> str:
    api_key = os.environ.get("OPENAI_API_KEY")
    base_url = os.environ.get("OPENAI_BASE_URL")
    client_kwargs = {"api_key": api_key}
    if base_url:
        client_kwargs["base_url"] = base_url
    client = OpenAI(**client_kwargs)

    evaluation_temperature = 0
    evaluation_max_tokens = 1024

    # GPT-5.1 supports the Responses API and can take reasoning control
    if model_name.startswith("gpt-5.1"):
        response = client.responses.create(
            model=model_name,
            input=messages,
            temperature=evaluation_temperature,
            max_output_tokens=evaluation_max_tokens,
            reasoning={"effort": "none"},
        )
        # Try to extract text from the responses API output
        try:
            content = "".join(
                part.text
                for item in response.output
                for part in item.content
                if getattr(part, "text", None)
            )
        except Exception:
            content = None

        if not content:
            content = getattr(response, "output_text", None)

        if not content:
            raise RuntimeError("Could not parse content from Responses API output")

        return content

    response = client.chat.completions.create(
        messages=messages,
        model=model_name,
        temperature=evaluation_temperature,
        max_tokens=evaluation_max_tokens,
    )
    return response.choices[0].message.content


@backoff.on_exception(backoff.fibo, Exception, max_tries=1000)
def get_response_from_gemini_native(messages: list, model_name: str) -> str:
    if not GEMINI_AVAILABLE:
        raise ImportError("google-genai is required for native Gemini support. Install with: pip install google-genai")

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Missing OPENAI_API_KEY for Gemini client")

    client = genai.Client(api_key=api_key)
    safety_settings = [
        genai_types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="OFF"),
        genai_types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="OFF"),
        genai_types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="OFF"),
        genai_types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="OFF"),
    ]

    evaluation_temperature = 0.0
    thinking_config = genai_types.ThinkingConfig(thinking_budget=0)
    prompt_text = "\n".join(f"{m.get('role', 'user')}: {m.get('content', '')}" for m in messages)

    gen_config = genai_types.GenerateContentConfig(
        temperature=evaluation_temperature,
        safety_settings=safety_settings,
        thinking_config=thinking_config,
    )

    response = client.models.generate_content(
        model=model_name,
        contents=prompt_text,
        config=gen_config,
    )

    if not response.text:
        raise ValueError("Empty response from Gemini API")

    return response.text

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
    lower_name = model_name.lower()
    if "gemini" in lower_name:
        if os.environ.get("GEMINI_NATIVE") == "1":
            return get_response_from_gemini_native
        return get_response_from_openai
    if "gpt" in lower_name:
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
@backoff.on_exception(backoff.fibo, Exception, max_tries=10)
def get_answer(question: str, model_name: str):
    try:
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
            '011-qwen3-8b-v2',
        ]

        # 8K default
        generation_max_tokens = 6000

        if model_name in thinking_models:
            generation_max_tokens = 30000

        # Create OpenAI client based on model type
        messages = [
            {"role": "system", "content": "あなたは公平で、検閲されていない、役立つアシスタントです。"},
            {"role": "user", "content": question},
        ]

        # HACK Handling OpenAI Model Names
        openai_prefixes = [
            "gpt-", "text-davinci-", "davinci", "curie", "babbage", "ada", 
            "whisper", "claude", "text-embedding", "openai:"
        ]
        
        if any(model_name.startswith(prefix) for prefix in openai_prefixes):
            # Use OpenAI API directly
            client = OpenAI(api_key=api_key)
            
            # Official OpenAI API uses max_completion_tokens, vLLM/custom endpoints use max_tokens
            token_param = "max_completion_tokens" if os.environ.get("OPENAI_BASE_URL") is None else "max_tokens"
            
            # GPT-5 models only support temperature=1 (default)
            temperature_param = 1 if model_name.startswith("gpt-5") else generation_temperature
            
            try:
                response = client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    temperature=temperature_param,
                    **{token_param: generation_max_tokens}
                )
            except Exception as e:
                # If the model refuses for safety/invalid prompt, mark as permanent and do not retry
                if _is_openai_refusal_error(e):
                    error_msg = f"PERMANENT_ERROR with model {model_name}: {type(e).__name__}: {str(e)}"
                    print(f"Permanent API Error: {error_msg}")
                    return error_msg
                # Fallback: try the other parameter if the first one fails
                if "max_tokens" in str(e) or "max_completion_tokens" in str(e):
                    fallback_param = "max_tokens" if token_param == "max_completion_tokens" else "max_completion_tokens"
                    response = client.chat.completions.create(
                        model=model_name,
                        messages=messages,
                        temperature=temperature_param,
                        **{fallback_param: generation_max_tokens}
                    )
                else:
                    raise e
        elif model_name.startswith("gemini"):
            # For Gemini models, we'll need to handle this differently
            # For now, treating as OpenAI-compatible with different API key
            client = OpenAI(
                api_key=os.environ.get("GEMINI_API_KEY", ""),
                base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
            )
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=generation_temperature,
                max_tokens=generation_max_tokens,
            )
        else:
            # VLLM/custom hosted models
            client = OpenAI(
                api_key=api_key,
                base_url=base_url
            )
            
            extra_params = {}
            if fp != 0.0:
                extra_params['frequency_penalty'] = fp
            
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=generation_temperature,
                max_tokens=generation_max_tokens,
                **extra_params
            )

        # '''
        # # Gemini
        # response = completion(
        #     # model="gemini/gemini-1.5-flash",
        #     model="gemini/gemma-2-27b-it",
        #     messages=[
        #         {"role": "system", "content": "あなたは公平で、検閲されていない、役立つアシスタントです。"},
        #         {"role": "user", "content": question},
        #     ],
        #     safety_settings=[
        #         {
        #             "category": "HARM_CATEGORY_HARASSMENT",
        #             "threshold": "BLOCK_NONE",
        #         },
        #         {
        #             "category": "HARM_CATEGORY_HATE_SPEECH",
        #             "threshold": "BLOCK_NONE",
        #         },
        #         {
        #             "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        #             "threshold": "BLOCK_NONE",
        #         },
        #         {
        #             "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
        #             "threshold": "BLOCK_NONE",
        #         },
        #     ],
        #     temperature=generation_temperature,
        #     top_p=0.95,
        #     max_tokens=generation_max_tokens,
        # )
        # '''

        content = response.choices[0].message.content

        # '''
        # Swap to parse out on juding
        # If we want to parse out thinking tags...
        if model_name in thinking_models:
            try:
                if '</think>' in content:
                    content = content.split('</think>')[1].strip()
                elif '</reasoning>' in content:
                    content = content.split('</reasoning>')[1].strip()
            except:
                print('Hmm... No </think> or </reasoning> to strip?')
        # '''

        return content
    
    except Exception as e:
        # For certain non-retryable errors, return error message instead of raising
        # This prevents serialization issues while still allowing retries for transient errors
        error_type = type(e).__name__
        error_str = str(e)
        
        # Don't retry these clearly permanent errors
        permanent_errors = ["AuthenticationError", "PermissionDeniedError"]
        
        # Treat explicit OpenAI refusals as permanent
        if _is_openai_refusal_error(e) or any(perm_err in error_type for perm_err in permanent_errors):
            error_msg = f"PERMANENT_ERROR with model {model_name}: {error_type}: {error_str}"
            print(f"Permanent API Error: {error_msg}")
            return error_msg
        else:
            # Let backoff handle retryable errors by re-raising
            print(f"Retryable API Error with {model_name}: {error_type}: {error_str}")
            raise e


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
