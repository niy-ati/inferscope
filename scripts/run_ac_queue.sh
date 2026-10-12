#!/bin/bash
cd /mnt/c/Users/niyat/content/inferscope
source ~/inferscope/.venv-vllm/bin/activate
run() { echo "=== $*"; python -u -m inferscope.bench "$@" 2>&1 | grep -E ' c=|Error|error:' | cut -c1-420; }
run kv --variant qwen3-1.7b-bf16 --concurrency 8,16 --configs prefix_on,flashinfer_bf16_kv,flashinfer_fp8_kv
run kv --variant qwen3-1.7b-bf16 --concurrency 8,16 --configs prefix_off,prefix_on --engine sglang
echo QUEUE_DONE
