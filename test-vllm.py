import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams

num_gpus_per_model = 8
cuda_visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES', '')
gpu_ids = cuda_visible_devices.split(',')
gpu_count = len([gpu for gpu in gpu_ids if gpu])
if gpu_count:
    num_gpus_per_model = gpu_count



model_path = 'tokyotech-llm/Swallow-70b-instruct-v0.1'
tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, trust_remote_code=True)
llm = LLM(model=model_path, tensor_parallel_size=num_gpus_per_model, trust_remote_code=True)

chat = []
chat.append({'role': 'system', 'content': 'You are a helpful assistant'})
chat.append({'role': 'user', 'content': 'tell me a joke'})

input_ids = tokenizer.apply_chat_template(chat, add_generation_prompt=True)

# Generate w/ vLLM
max_new_token=2048
temperature=0.5
top_p=0.95
repetition_penalty=1.05

sampling_params = SamplingParams(
  max_tokens=max_new_token,
  temperature=temperature,
  top_p=top_p,
  repetition_penalty=repetition_penalty,
)
outputs = llm.generate(prompt_token_ids=[input_ids], sampling_params=sampling_params, use_tqdm=False)
output = outputs[0].outputs[0].text.strip()
print(output)
