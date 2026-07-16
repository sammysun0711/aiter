#!/bin/bash 
input_tokens=1024
output_tokens=4

mkdir -p profile_log
REQUEST_RATE=(1) 

for request_rate in ${REQUEST_RATE[@]}; do 
    echo "Running with request rate:${request_rate}" 
    vllm bench serve \
        --backend vllm \
        --model /root/workspace/models/Llama-2-70b-hf \
        --host 0.0.0.0 \
        --port 30000 \
        --dataset-name sharegpt \
        --dataset-path /root/workspace/ShareGPT_V3_unfiltered_cleaned_split_no_imsorry.json \
        --num-prompts 4 \
        --request-rate ${request_rate} \
        --profile \
        2>&1 | tee "./profile_log/bf16_profile_run_client_isl_${input_tokens}_osl_${output_tokens}_rate${request_rate}.log"
done 