model=shisa-ai/shisa-llama3-8b-v1
volume=/fsx/user02/.cache/huggingface/hub

docker run --gpus all --shm-size 1g -p 8080:80 -v $volume:/data ghcr.io/huggingface/text-generation-inference:2.0 --model-id $model
