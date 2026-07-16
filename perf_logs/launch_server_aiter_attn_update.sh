#!/bin/bash
export GPU_ARCHS="gfx942"
export VLLM_ROCM_USE_AITER=1
export VLLM_ROCM_USE_AITER_PAGED_ATTN=1
export VLLM_ROCM_USE_AITER_LINEAR=1
export VLLM_ROCM_USE_AITER_RMSNORM=1
export VLLM_ROCM_USE_AITER_MHA=1
export VLLM_ROCM_USE_AITER_UNIFIED_ATTENTION=0
export VLLM_ROCM_USE_SKINNY_GEMM=0
export VLLM_ROCM_QUICK_REDUCE_QUANTIZATION=INT4
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export TRITON_HIP_USE_ASYNC_COPY=1
export TRITON_HIP_USE_BLOCK_PINGPONG=1
export TRITON_HIP_ASYNC_FAST_SWIZZLE=1
export TRITON_HIP_PRESHUFFLE_SCALES=1
#      --num-scheduler-steps 10 \

python3 -m vllm.entrypoints.openai.api_server \
      --profiler-config '{"profiler": "torch", "torch_profiler_dir": "./vllm_profile_result_update"}' \
      --attention-config '{"backend": "ROCM_AITER_FA"}' \
      --model /root/workspace/models/Llama-2-70b-hf \
      --port 30000 \
      --host 0.0.0.0 \
      --tensor-parallel-size 8 \
      --max-model-len 4096 \
      --trust-remote-code \
      --gpu-memory-utilization 0.8 \
      --max-num-seqs 256 \
      --prefix-caching-hash-algo xxhash \
      --async-scheduling \
      2>&1 | tee ./vllm_server_llama2_update.log
