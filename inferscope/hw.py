"""Hardware ceilings: datasheet numbers and what this card actually sustains.

A laptop GPU rarely reaches its datasheet (power limits, clocks, memory
controller efficiency), so the performance model can use either the spec
or numbers measured here with two microbenchmarks:
  - memory bandwidth: a large device-to-device copy (reads + writes), and
    a read-only reduction (what streaming weights during decode looks like)
  - dense fp16 tensor-core throughput: a large square matmul
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import asdict, dataclass

import torch

# Datasheet memory bandwidth. RTX 3050 6GB Laptop: GA107, 96-bit GDDR6 @
# 14 Gbps -> 168 GB/s.
SPECS = {
    "NVIDIA GeForce RTX 3050 6GB Laptop GPU": {"bw_gbs": 168.0},
}
# Dense fp16 tensor-core FLOPs per clock per SM with fp32 accumulation (what
# PyTorch and vLLM use). Consumer Ampere (GA10x) runs fp32-accumulate at half
# the fp16-accumulate rate: 1024 / 2 = 512 (RTX 3090: 82 SMs x 1.695 GHz x
# 512 = 71 TFLOPS, NVIDIA's published fp32-accumulate figure).
TENSOR_FLOPS_PER_CLK_SM = {(8, 6): 512}


@dataclass
class Hardware:
    name: str
    vram_gb: float
    spec_bw_gbs: float | None
    spec_fp16_tflops: float | None
    meas_bw_gbs: float | None = None
    meas_read_bw_gbs: float | None = None
    meas_fp16_tflops: float | None = None
    sm_clock_mhz: int | None = None
    power_limit_w: float | None = None


def _time_ms(fn, iters: int) -> float:
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def measure_bandwidth(gib: float = 1.0) -> float:
    n = int(gib * 2**30) // 2
    src = torch.empty(n, dtype=torch.float16, device="cuda")
    dst = torch.empty_like(src)
    ms = _time_ms(lambda: dst.copy_(src), 20)
    return 2 * src.numel() * 2 / (ms / 1e3) / 1e9  # bytes read + written


def measure_read_bandwidth(gib: float = 1.0) -> float:
    n = int(gib * 2**30) // 2
    src = torch.ones(n, dtype=torch.float16, device="cuda")
    out = torch.empty((), dtype=torch.float32, device="cuda")
    ms = _time_ms(lambda: torch.sum(src, dim=0, dtype=torch.float32, out=out), 20)
    return src.numel() * 2 / (ms / 1e3) / 1e9


def measure_fp16_tflops(n: int = 4096) -> float:
    a = torch.randn(n, n, dtype=torch.float16, device="cuda")
    b = torch.randn(n, n, dtype=torch.float16, device="cuda")
    ms = _time_ms(lambda: a @ b, 30)
    return 2 * n**3 / (ms / 1e3) / 1e12


def _smi(field: str) -> str:
    try:
        return subprocess.run(["nvidia-smi", f"--query-gpu={field}", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10).stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        return ""


def probe(measure: bool = True) -> Hardware:
    name = torch.cuda.get_device_name(0)
    props = torch.cuda.get_device_properties(0)
    spec = SPECS.get(name, {})
    clock = _smi("clocks.max.sm")
    per_clk = TENSOR_FLOPS_PER_CLK_SM.get((props.major, props.minor))
    peak = props.multi_processor_count * per_clk * float(clock) * 1e6 / 1e12 if per_clk and clock.replace(".", "").isdigit() else None
    hw = Hardware(
        name=name,
        vram_gb=props.total_memory / 2**30,
        spec_bw_gbs=spec.get("bw_gbs"),
        spec_fp16_tflops=round(peak, 2) if peak else None,
    )
    if measure:
        # Laptop clocks move with temperature: report the median of 5 runs.
        med = lambda f: sorted(f() for _ in range(5))[2]
        hw.meas_bw_gbs = round(med(measure_bandwidth), 1)
        hw.meas_read_bw_gbs = round(med(measure_read_bandwidth), 1)
        hw.meas_fp16_tflops = round(med(measure_fp16_tflops), 2)
    power = _smi("power.limit")
    hw.sm_clock_mhz = int(float(clock)) if clock.replace(".", "").isdigit() else None
    hw.power_limit_w = float(power) if power.replace(".", "").isdigit() else None
    return hw


if __name__ == "__main__":
    print(json.dumps(asdict(probe()), indent=2))
