#!/bin/bash
cd /mnt/c/Users/niyat/content/inferscope
source ~/inferscope/.venv-vllm/bin/activate
export VLLM_USE_FLASHINFER_SAMPLER=0
run() { echo "=== $*"; python -u -m inferscope.bench "$@" 2>&1 | grep -E ' c=|^\{|Error|error:' | cut -c1-500; }
run kv --variant qwen3-1.7b-bf16 --concurrency 8,16 --configs prefix_on_mem86 --engine sglang
run quality --variant qwen3-1.7b-bf16
run quality --variant qwen3-4b-awq
echo QUEUE_DONE
