"""Tensor parallelism on Kaggle's free 2x T4: Qwen3 on vLLM, TP=1 vs TP=2,
replaying the real PulseLoop trace with the same load generator as the
laptop runs.

Runs as a Kaggle script kernel. The repo bundle (inferscope/ + the trace) is
attached as a private dataset. Set the notebook's accelerator to "GPU T4 x2":
Kaggle does not attach a GPU to API-started runs on this account.

  Qwen3-4B fp16 (8 GB)   fits one T4: TP=1 vs TP=2 shows what splitting costs
  Qwen3-8B fp16 (16 GB)  does not fit one 15 GB T4: TP=2 is what makes it servable
"""

import glob
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


def sh(cmd: str) -> int:
    print("$", cmd, flush=True)
    return subprocess.run(cmd, shell=True).returncode


gpus = subprocess.run("nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv,noheader",
                      shell=True, capture_output=True, text=True).stdout.strip().splitlines()
print("GPUs:", gpus, flush=True)
if len(gpus) < 2:
    sys.exit("Needs 2 GPUs. In the notebook's settings set Accelerator = GPU T4 x2, then Save Version.")

found = glob.glob("/kaggle/input/**/inferscope/bench.py", recursive=True)
if not found:  # the dataset may mount the uploaded folder still zipped
    for z in glob.glob("/kaggle/input/**/*.zip", recursive=True):
        shutil.unpack_archive(z, "/kaggle/working/bundle")
    found = glob.glob("/kaggle/working/bundle/**/inferscope/bench.py", recursive=True)
bundle = Path(found[0]).parent.parent
work = Path("/kaggle/working/inferscope")
shutil.copytree(bundle, work, dirs_exist_ok=True)
os.chdir(work)

sh("pip install -q vllm httpx jsonschema huggingface_hub")
ok = subprocess.run([sys.executable, "-c", "import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.version.cuda)"]).returncode == 0
if not ok:  # the newest vLLM wheel may need a newer driver than Kaggle's
    sh("pip install -q 'vllm==0.11.0'")

from huggingface_hub import snapshot_download  # noqa: E402

models = Path.home() / "models"
for name in ("Qwen3-4B", "Qwen3-8B"):
    snapshot_download(f"Qwen/{name}", local_dir=models / name, allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model"])

os.environ.update({
    "INFERSCOPE_VLLM_EXTRA": "--dtype float16 --disable-custom-all-reduce",  # Turing: no bf16; custom all-reduce needs NVLink/P2P
    "INFERSCOPE_THROTTLE_MHZ": "0",        # the laptop's clock threshold doesn't apply to T4s
    "INFERSCOPE_START_TIMEOUT": "2400",    # 4 vCPUs: compiling CUDA graphs is slow
    "VLLM_USE_FLASHINFER_SAMPLER": "0",
})

runs = [("hw", ["hw"]),
        ("4b-tp1", ["perf", "--variant", "qwen3-4b-fp16", "--concurrency", "1,4,16,32", "--tp", "1", "--tag=-t4-tp1"]),
        ("4b-tp2", ["perf", "--variant", "qwen3-4b-fp16", "--concurrency", "1,4,16,32", "--tp", "2", "--tag=-t4-tp2"]),
        ("8b-tp2", ["perf", "--variant", "qwen3-8b-fp16", "--concurrency", "1,4,16,32", "--tp", "2", "--tag=-t4-tp2"])]
status = {}
for name, args in runs:
    rc = subprocess.run([sys.executable, "-u", "-m", "inferscope.bench", *args]).returncode
    if rc != 0 and name != "hw":  # retry once on the Triton attention backend (FlashAttention needs sm_80+)
        os.environ["INFERSCOPE_VLLM_EXTRA"] += " --attention-backend TRITON_ATTN"
        rc = subprocess.run([sys.executable, "-u", "-m", "inferscope.bench", *args]).returncode
    status[name] = rc
    print("STATUS", name, rc, flush=True)

out = Path("/kaggle/working/results_t4")
out.mkdir(exist_ok=True)
for f in glob.glob("results/*.json"):
    shutil.copy(f, out)
shutil.copy("results/hw.json", out / "hw_t4.json")
for f in glob.glob("results/logs/*.log"):
    shutil.copy(f, out)
(out / "status.json").write_text(json.dumps({"gpus": gpus, "status": status}, indent=2))
print("DONE", status)
