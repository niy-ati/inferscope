"""Start and stop inference servers, and read what they report about memory.

Every engine runs as its own process with an OpenAI-compatible endpoint, so
the load generator treats them identically. GPU memory is sampled with
nvidia-smi while the benchmark runs.
"""

from __future__ import annotations

import os
import re
import shutil
import signal
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import httpx


def server_env(cuda_home: Path | None) -> dict[str, str]:
    """No system CUDA toolkit or g++ here. FlashInfer JIT-compiles kernels at
    startup (FP8 KV cache attention in vLLM; SGLang always), and its CCCL
    headers insist that nvcc matches the CUDA headers the engine's torch was
    built with: CUDA 13.0 for vLLM (pip wheels), 12.8 for SGLang (conda-forge).
    The sampler stays on vLLM's PyTorch path so every run uses the same one."""
    env = {**os.environ, "PYTHONUNBUFFERED": "1", "VLLM_USE_FLASHINFER_SAMPLER": "0", "VLLM_SERVER_DEV_MODE": "1"}
    if cuda_home and (cuda_home / "bin" / "nvcc").exists():
        env["CUDA_HOME"] = str(cuda_home)
        env["PATH"] = f"{cuda_home / 'bin'}:{env.get('PATH', '')}"
        if (cuda_home / "targets/x86_64-linux/include").exists():  # conda layout
            env["CPATH"] = f"{cuda_home / 'targets/x86_64-linux/include'}:{env.get('CPATH', '')}"
    # The system gcc has no C++ front end (cc1plus). Use a conda-forge
    # toolchain when there is one: FlashInfer passes CC to nvcc as -ccbin.
    if not shutil.which("g++") and "CXX" not in os.environ:
        for gxx in sorted(Path.home().glob("mamba/envs/*/bin/x86_64-conda-linux-gnu-g++")):
            env["CXX"], env["CC"] = str(gxx), str(gxx.with_name("x86_64-conda-linux-gnu-gcc"))
            # SGLang's own JIT builds ignore CC/CXX; nvcc honours this for every call.
            env["NVCC_PREPEND_FLAGS"] = f"-ccbin {gxx} {env.get('NVCC_PREPEND_FLAGS', '')}".strip()
            break
    # SGLang's sgl-kernel links libnuma, which this system lacks; conda-forge ships it.
    for numa in sorted(Path.home().glob("mamba/envs/*/lib/libnuma.so.1"))[:1]:
        env["LD_LIBRARY_PATH"] = f"{numa.parent}:{env.get('LD_LIBRARY_PATH', '')}"
    return env


@dataclass
class Server:
    engine: str
    cmd: list[str]
    port: int
    log_path: Path
    proc: subprocess.Popen | None = None
    started_s: float = 0.0
    info: dict = field(default_factory=dict)
    cuda_home: Path | None = None

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def start(self, timeout: float = 900) -> "Server":
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        log = open(self.log_path, "w")
        t0 = time.time()
        self.proc = subprocess.Popen(self.cmd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True, env=server_env(self.cuda_home))
        while time.time() - t0 < timeout:
            if self.proc.poll() is not None:
                raise RuntimeError(f"{self.engine} exited early, see {self.log_path}:\n" + self.log_path.read_text()[-3000:])
            try:
                if httpx.get(self.url + "/v1/models", timeout=2).status_code == 200:
                    self.started_s = time.time() - t0
                    self.info = parse_log(self.engine, self.log_path.read_text())
                    return self
            except httpx.HTTPError:
                pass
            time.sleep(2)
        self.stop()
        raise TimeoutError(f"{self.engine} did not come up in {timeout}s")

    def reset_prefix_cache(self) -> None:
        """Start a measurement with a cold prefix cache."""
        path = "/flush_cache" if self.engine == "sglang" else "/reset_prefix_cache"  # vLLM: dev-mode endpoint
        httpx.post(self.url + path, timeout=30).raise_for_status()

    def prefix_counters(self) -> tuple[float, float]:
        """(queried, hit) prompt tokens, from vLLM's Prometheus metrics."""
        text = httpx.get(self.url + "/metrics", timeout=10).text
        def total(name: str) -> float:
            return sum(float(line.rsplit(" ", 1)[1]) for line in text.splitlines() if line.startswith(name + "{") or line.startswith(name + " "))
        return total("vllm:prefix_cache_queries_total"), total("vllm:prefix_cache_hits_total")

    def model_id(self) -> str:
        return httpx.get(self.url + "/v1/models", timeout=10).json()["data"][0]["id"]

    def stop(self) -> None:
        if self.proc and self.proc.poll() is None:
            os.killpg(self.proc.pid, signal.SIGTERM)
            try:
                self.proc.wait(60)
            except subprocess.TimeoutExpired:
                os.killpg(self.proc.pid, signal.SIGKILL)
        _wait_gpu_free()

    def __enter__(self) -> "Server":
        return self.start()

    def __exit__(self, *exc: object) -> None:
        self.stop()


def vllm(model: str, port: int, log: Path, *, max_len: int = 8192, util: float = 0.82, prefix_cache: bool = True,
         quantization: str | None = None, kv_dtype: str = "auto", max_seqs: int = 32, eager: bool = False, tp: int = 1, extra: list[str] | None = None) -> Server:
    cmd = ["vllm", "serve", model, "--port", str(port), "--max-model-len", str(max_len), "--gpu-memory-utilization", str(util),
           "--max-num-seqs", str(max_seqs), "--kv-cache-dtype", kv_dtype, "--enable-prompt-tokens-details",
           "--enable-prefix-caching" if prefix_cache else "--no-enable-prefix-caching"]
    if quantization:
        cmd += ["--quantization", quantization]
    if eager:
        cmd += ["--enforce-eager"]
    if tp > 1:  # tensor parallel across GPUs (multi-GPU hosts; not run on the 1-GPU laptop)
        cmd += ["--tensor-parallel-size", str(tp)]
    return Server("vllm", cmd + (extra or []), port, log, cuda_home=_toolkit("INFERSCOPE_CUDA_VLLM", "~/inferscope/cuda130/nvidia/cu13"))


def _toolkit(var: str, default: str) -> Path | None:
    p = Path(os.path.expanduser(os.getenv(var, default)))
    return p if p.exists() else None


SGLANG_PY = os.path.expanduser(os.getenv("INFERSCOPE_SGLANG_PY", "~/inferscope/.venv-sglang/bin/python"))


def sglang(model: str, port: int, log: Path, *, max_len: int = 8192, mem: float = 0.82, prefix_cache: bool = True,
           quantization: str | None = None, kv_dtype: str = "auto", max_seqs: int = 32, tp: int = 1, extra: list[str] | None = None) -> Server:
    """SGLang in its own venv (its pinned torch differs from vLLM's). Prefix
    reuse is RadixAttention; --enable-cache-report puts cached_tokens in usage."""
    cmd = [SGLANG_PY, "-m", "sglang.launch_server", "--model-path", model, "--port", str(port), "--context-length", str(max_len),
           "--mem-fraction-static", str(mem), "--max-running-requests", str(max_seqs), "--kv-cache-dtype", kv_dtype,
           "--enable-cache-report", "--enable-metrics"]
    if not prefix_cache:
        cmd += ["--disable-radix-cache"]
    if tp > 1:
        cmd += ["--tp-size", str(tp)]
    if quantization:
        cmd += ["--quantization", quantization]
    return Server("sglang", cmd + (extra or []), port, log, cuda_home=_toolkit("INFERSCOPE_CUDA_SGLANG", "~/mamba/envs/cuda128"))


def llamacpp(gguf: str, port: int, log: Path, *, ctx: int = 8192, parallel: int = 8, binary: str = "llama-server") -> Server:
    cmd = [binary, "-m", gguf, "--port", str(port), "-c", str(ctx * parallel), "-np", str(parallel), "-ngl", "999", "-fa", "on", "--jinja"]
    return Server("llamacpp", cmd, port, log)


def parse_log(engine: str, text: str) -> dict:
    """Numbers the engine prints at startup (vLLM: weights, KV cache)."""
    info: dict = {}
    if engine == "vllm":
        if m := re.search(r"Model loading took ([\d.]+) GiB", text):
            info["weights_gib"] = float(m.group(1))
        if m := re.search(r"GPU KV cache size: ([\d,]+) tokens", text):
            info["kv_cache_tokens"] = int(m.group(1).replace(",", ""))
        if m := re.search(r"Available KV cache memory: ([\d.]+) GiB", text):
            info["kv_cache_gib"] = float(m.group(1))
        if m := re.search(r"Maximum concurrency for ([\d,]+) tokens per request: ([\d.]+)x", text):
            info["max_concurrency"] = float(m.group(2))
        if m := re.search(r"Graph capturing finished in ([\d.]+) secs?, took ([\d.]+) GiB", text):
            info["cudagraph_gib"] = float(m.group(2))
    elif engine == "sglang":
        if m := re.search(r"Load weight end.*?mem usage=([\d.]+) GB", text):
            info["weights_gib"] = float(m.group(1))
        if m := re.search(r"max_total_num_tokens=(\d+)", text):
            info["kv_cache_tokens"] = int(m.group(1))
        if m := re.search(r"KV Cache is allocated\..*?K size: ([\d.]+) GB, V size: ([\d.]+) GB", text):
            info["kv_cache_gib"] = round(float(m.group(1)) + float(m.group(2)), 2)
    return info


def gpu_used_mib() -> int:
    if not shutil.which("nvidia-smi"):  # e.g. AMD/ROCm: no NVIDIA telemetry
        return 0
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout
    return int(out.strip().splitlines()[0])


def _wait_gpu_free(limit_mib: int = 400, timeout: float = 60) -> None:
    t0 = time.time()
    while time.time() - t0 < timeout and gpu_used_mib() > limit_mib:
        time.sleep(1)


class GpuSampler:
    """Peak GPU memory, utilisation, clocks and power while a benchmark runs.
    Clocks and throttle reasons matter on a laptop: the driver lowers clocks
    at light load and under power or thermal limits."""

    FIELDS = "memory.used,utilization.gpu,clocks.sm,clocks.mem,power.draw,temperature.gpu,clocks_event_reasons.active"

    def __init__(self, every: float = 0.25):
        self.every, self.peak_mib, self.rows, self._stop = every, 0, [], threading.Event()

    def _run(self) -> None:
        if not shutil.which("nvidia-smi"):  # e.g. AMD/ROCm: latencies still measured, no GPU telemetry
            return
        while not self._stop.is_set():
            out = subprocess.run(["nvidia-smi", f"--query-gpu={self.FIELDS}", "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout
            f = [x.strip() for x in out.strip().splitlines()[0].split(",")]
            mem, util, sm, memclk = (int(float(x)) for x in f[:4])
            power = float(f[4]) if f[4].replace(".", "", 1).isdigit() else 0.0
            self.peak_mib = max(self.peak_mib, mem)
            self.rows.append((util, sm, memclk, power, int(f[5]), int(f[6], 16)))
            time.sleep(self.every)

    def __enter__(self) -> "GpuSampler":
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        self._t.join()

    @property
    def mean_util(self) -> float:
        return sum(r[0] for r in self.rows) / len(self.rows) if self.rows else 0.0

    def summary(self) -> dict:
        if not self.rows:
            return {}
        n = len(self.rows)
        busy = [r for r in self.rows if r[0] > 50] or self.rows  # ignore idle gaps between requests
        reasons = 0
        for r in busy:
            reasons |= r[5]
        return {"gpu_util_mean": round(sum(r[0] for r in self.rows) / n, 1),
                "sm_clock_mean_mhz": round(sum(r[1] for r in busy) / len(busy)),
                "sm_clock_min_mhz": min(r[1] for r in busy),
                "mem_clock_mean_mhz": round(sum(r[2] for r in busy) / len(busy)),
                "power_mean_w": round(sum(r[3] for r in busy) / len(busy), 1),
                "temp_max_c": max(r[4] for r in self.rows),
                "throttle_reasons": hex(reasons)}
