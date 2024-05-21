import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_path = '/fsx/user02//axolotl/outputs/lr-2e6'
# model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="auto")

'''
# normal
# 24.2s
model = AutoModelForCausalLM.from_pretrained(
            model_path, 
            low_cpu_mem_usage=True,
            device_map="auto",
        )
'''

'''
# 23.5s
model = AutoModelForCausalLM.from_pretrained(
            model_path, 
            device_map="auto",
        )
'''

'''
# Added torch, not so different
# 23.2s
model = AutoModelForCausalLM.from_pretrained(
            model_path, 
            device_map="auto",
        )
'''

# 8X 16.5s , 1X 11.1s, 2X 12.5s, 4X 13.3s
model = AutoModelForCausalLM.from_pretrained(
            model_path, 
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )
