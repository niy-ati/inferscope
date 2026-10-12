#!/bin/bash
# Distributed serving and AMD runs. These need hardware the laptop doesn't
# have (2+ GPUs, or an AMD Instinct GPU), so this script is for a rented
# cloud machine. No results from it are in this repo yet.
#
# Same trace, same load generator, same output lengths as the laptop runs,
# so the numbers are directly comparable.
#
#   NVIDIA, 2+ GPUs:  bash scripts/cloud_multi_gpu.sh nvidia
#   AMD (ROCm):       bash scripts/cloud_multi_gpu.sh amd
#
# Needs: the private trace at data/trace.jsonl (copy it over), and the model
# under ~/models/. Qwen3-8B is big enough that tensor parallelism matters.
set -euo pipefail
cd "$(dirname "$0")/.."
platform=${1:?usage: cloud_multi_gpu.sh nvidia|amd}
variant=qwen3-8b-bf16   # model files expected at ~/models/Qwen3-8B

if [ "$platform" = amd ]; then
  # vLLM's ROCm build ships as a container; GPU telemetry (nvidia-smi) is
  # skipped automatically, latencies are still recorded.
  pip install -e . --no-deps
fi

for tp in 1 2; do
  python -m inferscope.bench perf --variant "$variant" --concurrency 1,4,16,32 --tp $tp --tag "-$platform-tp$tp"
done
# KV-cache findings, re-checked on datacenter hardware
python -m inferscope.bench kv --variant "$variant" --concurrency 16,32 --configs prefix_off,prefix_on
