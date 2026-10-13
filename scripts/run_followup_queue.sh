#!/bin/bash
# After run_remaining_queue.sh: redo the GPTQ-generic speed sweep (the first
# one ran on battery), then the reference-loss check (untimed). Git Bash on
# Windows, like the other queues.
step() {
  until powershell -NoProfile -c "(Get-CimInstance -Namespace root/wmi -ClassName BatteryStatus).PowerOnline" | grep -q True; do
    echo "waiting for AC power $(date +%H:%M)"; sleep 60
  done
  echo "=== $* ($(date +%H:%M))"
  wsl -e bash -lc "cd /mnt/c/Users/niyat/content/inferscope && source ~/inferscope/.venv-vllm/bin/activate && VLLM_USE_FLASHINFER_SAMPLER=0 python -u -m inferscope.bench $* 2>&1 | grep -E ' c=|^\{|Error|error:' | cut -c1-600" | tr -d '\0'
}

step perf --variant qwen3-1.7b-w4a16-generic --concurrency 1,2,4,8,16
echo "=== reference loss check ($(date +%H:%M))"
wsl -e bash -lc "cd /mnt/c/Users/niyat/content/inferscope && source ~/inferscope/.venv-quant/bin/activate && python tools/reference_nll_check.py Qwen3-1.7B Qwen3-1.7B-W4A16-generic Qwen3-1.7B-W4A16-workload Qwen3-4B-AWQ 2>&1 | grep -E '^\{|Error'" | tr -d '\0'
echo QUEUE_DONE
