## Super Janky

# Terminal 0: Run vllm - so janky
# python -m vllm.entrypoints.openai.api_server --model shisa-ai/shisa-llama3-8b-v1 -tp 8 --max-model-len 8192

# Terminal 1: Edit llm_functions.py
# change repetition_penalty manually

# Terminal 2: Run this here

# rps=("1.05" "1.10" "1.12" "1.15" "1.18" "1.20" "1.22")
fps=("0.0" "0.5" "0.8")

# Generate Answers
for fp in "${fps[@]}"
do
	echo "> frequency penalty: $fp"
	echo "==="
	# Generate Judgments
	echo 'Judging...'
	python judge_answers.py -m shisa-ai/shisa-llama3-8b-v1.fp$fp -d shisa-ai/ja-mt-bench-1shot

	# Move
	# mv data/judgements/judge_gpt-4-turbo-preview/shisa-ai__ja-mt-bench-1shot/shisa-ai__shisa-llama3-8b-v1.json data/judgements/judge_gpt-4-turbo-preview/shisa-ai__ja-mt-bench-1shot/shisa-ai__shisa-llama3-8b-v1.rp-$rp.json
done


# Terminal 3: generate output
# python results_vizualization.py ; grep llama3 output.csv
