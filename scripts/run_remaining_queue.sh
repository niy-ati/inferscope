#!/bin/bash
# Remaining steps of run_completion_queue.sh after an interrupted session.
# and repeated runs for error bars.
#
# Runs from Git Bash on Windows (not inside WSL): it checks for AC power
# before every step, and WSL here cannot call Windows programs. Each step
# then runs inside WSL. On battery the GPU's power budget halves, which
# invalidates timing results.

step() {
  until powershell -NoProfile -c "(Get-CimInstance -Namespace root/wmi -ClassName BatteryStatus).PowerOnline" | grep -q True; do
    echo "waiting for AC power $(date +%H:%M)"; sleep 60
  done
  echo "=== $* ($(date +%H:%M))"
  wsl -e bash -lc "cd /mnt/c/Users/niyat/content/inferscope && source ~/inferscope/.venv-vllm/bin/activate && VLLM_USE_FLASHINFER_SAMPLER=0 python -u -m inferscope.bench $* 2>&1 | grep -E ' c=|^\{|Error|error:' | cut -c1-600" | tr -d '\0'
}

step perf --variant qwen3-1.7b-w4a16-workload --concurrency 1,2,4,8,16
step kv-repeat --variant qwen3-1.7b-bf16 --concurrency 16 --repeats 3 --configs pin_prefix_off,pin_prefix_on,pin_fi_bf16_kv,pin_fi_fp8_kv,pin_best
step kv-repeat --variant qwen3-1.7b-bf16 --concurrency 16 --repeats 3 --configs prefix_on_mem86 --engine sglang
echo QUEUE_DONE
