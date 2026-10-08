"""Start and stop inference servers, and read what they report about memory.

Every engine runs as its own process with an OpenAI-compatible endpoint, so
the load generator treats them identically. GPU memory is sampled with
nvidia-smi while the benchmark runs.
"""

from __future__ import annotations

import os
import re
import signal
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import httpx


@dataclass
class Server:
    engine: str
    cmd: list[str]
    port: int
    log_path: Path
    proc: subprocess.Popen | None = None
    started_s: float = 0.0
    info: dict = field(default_factory=dict)

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def start(self, timeout: float = 900) -> "Server":
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        log = open(self.log_path, "w")
        t0 = time.time()
        # FlashInfer's sampler JIT-compiles with nvcc, which this machine lacks; vLLM falls back to its PyTorch sampler.
        self.proc = subprocess.Popen(self.cmd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True, env={**os.environ, "PYTHONUNBUFFERED": "1", "VLLM_USE_FLASHINFER_SAMPLER": "0"})
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
         quantization: str | None = None, kv_dtype: str = "auto", max_seqs: int = 32, eager: bool = False, extra: list[str] | None = None) -> Server:
    cmd = ["vllm", "serve", model, "--port", str(port), "--max-model-len", str(max_len), "--gpu-memory-utilization", str(util),
           "--max-num-seqs", str(max_seqs), "--kv-cache-dtype", kv_dtype,
           "--enable-prefix-caching" if prefix_cache else "--no-enable-prefix-caching"]
    if quantization:
        cmd += ["--quantization", quantization]
    if eager:
        cmd += ["--enforce-eager"]
    return Server("vllm", cmd + (extra or []), port, log)


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
    return info


def gpu_used_mib() -> int:
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout
    return int(out.strip().splitlines()[0])


def _wait_gpu_free(limit_mib: int = 400, timeout: float = 60) -> None:
    t0 = time.time()
    while time.time() - t0 < timeout and gpu_used_mib() > limit_mib:
        time.sleep(1)


class GpuSampler:
    """Peak GPU memory and mean utilisation while a benchmark runs."""

    def __init__(self, every: float = 0.25):
        self.every, self.peak_mib, self.util, self._stop = every, 0, [], threading.Event()

    def _run(self) -> None:
        while not self._stop.is_set():
            out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,utilization.gpu", "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout
            mem, util = (int(x) for x in out.strip().splitlines()[0].split(","))
            self.peak_mib, _ = max(self.peak_mib, mem), self.util.append(util)
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
        return sum(self.util) / len(self.util) if self.util else 0.0
