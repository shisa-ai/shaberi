## Super Janky

source ~/.bashrc
mamba activate shaberi

# Terminal 0: Run vllm - so janky
# python -m vllm.entrypoints.openai.api_server -tp 8 --model /fsx/user02/axolotl/outputs/wd-0.05 --served-model-name 'shisa-ai/shisa-v1-llama3-8b.wd-0.05'
# python -m vllm.entrypoints.openai.api_server --model $model -tp 8 --max-model-len 8192 --served-model-name 'shisa-ai/shisa-swallowmx-13a47b-v1'
#
# 2 x 24GB version... (lowered gpu-memory-utilization since OOM)
# python -m vllm.entrypoints.openai.api_server --model shisa-ai/shisa-v1-phi3-14b --max-model-len 8192 -tp 2 --gpu-memory-utilization 0.85

# Terminal 2: Run this here

# rps=("1.05" "1.10" "1.12" "1.15" "1.18" "1.20" "1.22")
# fps=("0.0" "0.5" "0.8")
fps=("0.5")

# Generate Answers
for fp in "${fps[@]}"
do
	echo "> frequency penalty: $fp"
	echo "==="
	echo 'Generating answers...'
	OPENAI_BASE_URL='http://localhost:8000/v1' python generate_answers.py -m $1 -n 8 -fp $fp
	# Generate Judgments
	echo 'Judging...'
	python judge_answers.py -m $1
done

python results_vizualization.py ; cat output.csv
