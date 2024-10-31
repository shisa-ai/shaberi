# OPENAI_BASE_URL='http://localhost:8000/v1' python generate_answers.py --model_name 'augmxnt/shisa-gamma-7b-v1' --num_proc 8
# OPENAI_BASE_URL='http://localhost:8000/v1' python generate_answers.py --model_name 'augmxnt/shisa-7b-v1' --num_proc 8
# OPENAI_BASE_URL='http://localhost:8000/v1' python generate_answers.py --model_name 'shisa-ai/shisa-llama3-8b-v1' --num_proc 8
# OPENAI_BASE_URL='http://localhost:8000/v1' python generate_answers.py --model_name 'shisa-ai/shisa-yi1.5-9b-v1' --num_proc 8
# OPENAI_BASE_URL='http://localhost:8000/v1' python generate_answers.py --model_name 'shisa-ai/shisa-gemma-7b-v1' --num_proc 8
#OPENAI_BASE_URL='http://localhost:8000/v1' python generate_answers.py --model_name 'meta-llama/Meta-Llama-3-8B-Instruct' --num_proc 8
# OPENAI_BASE_URL='http://localhost:8000/v1' python generate_answers.py --model_name 'meta-llama/Meta-Llama-3-70B-Instruct' --num_proc 8
time python generate_answers.py --num_proc 40 --model_name nvidia/Llama-3.1-Nemotron-70B-Instruct-HF
