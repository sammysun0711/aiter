#!/bin/bash 

mkdir -p perf_log_bf16
REQUEST_RATE=(1 2 4 8 16 32 64 128 256 512) 

for request_rate in ${REQUEST_RATE[@]}; do 
    echo "Running with request rate:${request_rate}" 
    vllm bench serve \
        --backend vllm \
        --model /root/workspace/models/Llama-2-70b-hf \
        --host 0.0.0.0 \
        --port 30000 \
        --dataset-name sharegpt \
        --dataset-path /root/workspace/ShareGPT_V3_unfiltered_cleaned_split_no_imsorry.json \
        --num-prompts 1000 \
        --request-rate ${request_rate} \
        2>&1 | tee "./perf_log_bf16/bf16_run_client_rate${request_rate}.log"
done 