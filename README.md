# inferscope

How well can a real LLM workload be served on a 6 GB laptop GPU, and does a first-principles model predict what the engine actually does?

The workload is [PulseLoop](https://github.com/niy-ati/PulseLoop)'s production LLM traffic: planner, copywriting, captions, judge and video-director calls. It was rebuilt from stored runs and replayed against vLLM. Every number below comes from a file in [`results/`](results/).

## Hardware

NVIDIA RTX 3050 6 GB Laptop GPU (Ampere, sm_86), running under WSL2.

| | Datasheet | Measured |
|---|---|---|
| Memory bandwidth | 168 GB/s | 111 GB/s (copy), 117 GB/s (read) |
| fp16 tensor throughput | 21.5 TFLOPS | 12.2 TFLOPS |

The card sustains about 70% of its datasheet bandwidth and 57% of its compute. The performance model uses the measured ceilings, re-probed right before each run because laptop clocks drift with heat.

## Workload

74 requests from real PulseLoop runs, tokenized with the Qwen3 chat template.

| | min | p50 | p95 | max |
|---|---|---|---|---|
| Prompt tokens | 270 | 495 | 1,128 | 1,971 |
| Output tokens | 72 | 245 | 703 | 1,751 |

39% of prompt tokens sit in system prompts shared across requests, which is what makes prefix caching worth measuring.

## Results

### Serving on vLLM 0.31: Qwen3-1.7B (bf16) vs Qwen3-4B (AWQ, 4-bit)

32 requests per concurrency level, closed loop. Results: [`perf_qwen3-1.7b-bf16.json`](results/perf_qwen3-1.7b-bf16.json), [`perf_qwen3-4b-awq.json`](results/perf_qwen3-4b-awq.json).

| | Qwen3-1.7B bf16 | Qwen3-4B AWQ |
|---|---|---|
| Weights | 3.22 GiB | 2.50 GiB |
| KV cache that fits | 12,608 tokens | 10,496 tokens |
| Decode floor (weights ÷ 117 GB/s) | 29.5 ms/token | 22.9 ms/token |

Output throughput (tok/s) and median time per output token:

| Concurrency | bf16 tok/s | AWQ tok/s | bf16 TPOT | AWQ TPOT |
|---|---|---|---|---|
| 1 | 21.0 † | 38.0 | 45.8 ms † | 24.9 ms |
| 2 | 57.4 | 63.3 | 33.3 ms | 29.6 ms |
| 4 | 96.4 | 104.4 | 35.4 ms | 32.4 ms |
| 8 | 127.9 | 138.6 | 39.2 ms | 36.3 ms |
| 16 | 150.3 | 151.5 | 44.7 ms | 44.3 ms |

Time to first token, p50 / p95:

| Concurrency | bf16 | AWQ |
|---|---|---|
| 1 | 357 / 908 ms | 166 / 512 ms |
| 4 | 179 / 321 ms | 260 / 595 ms |
| 8 | 213 / 852 ms | 341 / 1,605 ms |
| 16 | 793 / 1,664 ms | 2,343 / 6,300 ms |

- **Decode is bandwidth-bound, and the model predicts it.** AWQ's single-request decode (24.9 ms) is within 9% of the weight-read floor. From 2 to 8 concurrent requests, per-token latency rises only about 20% while throughput more than doubles, because each extra sequence rides on the same weight read.
- **A 4-bit 4B model decodes faster than a bf16 1.7B one.** It reads fewer bytes per token: 2.5 vs 3.2 GiB. AWQ leads at every concurrency up to 8. At 16, both hit the same ceiling.
- **Prefill is where AWQ pays.** Its time to first token is higher from concurrency 4 up, because 4-bit weights are dequantized in compute-bound prefill. At 16, the smaller KV cache also makes requests queue: p95 is 6.3 s vs 1.7 s.
- **The KV cache is the hard limit.** About 2.5–3 worst-case requests fit at once. Past 8 concurrent requests, throughput gains little and first-token latency climbs, so on this card 8 is the sensible limit.

† **Unresolved:** bf16 at concurrency 1 decodes at 45.8 ms/token, 55% above its floor and slower than two requests at once. That shouldn't happen on a bandwidth-bound GPU. A longer warm-up removed the first-token stalls but not this. The leading suspect is the laptop GPU dropping clocks at light load. Run-to-run variance is also visible elsewhere: an earlier bf16 run got 7,984 tokens of KV cache, against 12,608 here, at the same memory setting. Repeated runs are planned before these numbers are treated as final.

Quality is not measured yet, so these tables say nothing about whether AWQ does PulseLoop's job as well. That's the next experiment.

### Fused MoE kernel (Triton)

One mixture-of-experts layer with real model shapes. The fused grouped-GEMM kernel is compared against a per-expert PyTorch loop and against the bandwidth bound.

| Model | Tokens | Fused | Naive | Speedup | % of bandwidth bound |
|---|---|---|---|---|---|
| Qwen3-30B-A3B | 1 | 0.64 ms | 56.7 ms | 89× | 92% |
| Qwen3-30B-A3B | 32 | 8.74 ms | 125.8 ms | 14× | 98% |
| Qwen3-30B-A3B | 512 | 11.55 ms | 110.1 ms | 9.5× | 84% |
| OLMoE-1B-7B | 1 | 0.83 ms | 26.0 ms | 31× | 94% |
| OLMoE-1B-7B | 32 | 6.31 ms | 55.8 ms | 8.8× | 97% |
| OLMoE-1B-7B | 512 | 9.04 ms | 59.0 ms | 6.5× | 74% |

Outputs match the reference within 0.1% relative error. Full table: [`results/moe.json`](results/moe.json).

### In progress

- Repeated runs, and the bf16 concurrency-1 anomaly
- FP8 / GPTQ W4A16 variants of Qwen3-1.7B on the same sweep
- GPTQ calibrated on generic text vs. on the workload itself
- Prefix caching on vs. off, in the trace's real request order
- Cost of JSON-schema guided decoding
- Quality per variant: WikiText-2 perplexity, likelihood of PulseLoop's reference answers, and JSON validity

## Layout

| Path | What it does |
|---|---|
| `inferscope/hw.py` | Datasheet vs. measured bandwidth and fp16 throughput |
| `inferscope/roofline.py` | Predicts prefill time, decode step time and KV capacity from model shape and hardware ceilings |
| `inferscope/loadgen.py` | Closed-loop replay of a trace against any OpenAI-compatible server |
| `inferscope/servers.py` | Starts and stops engines, samples GPU memory |
| `inferscope/quantize.py` | GPTQ W4A16 with two calibration sets |
| `inferscope/quality.py` | Perplexity, workload likelihood and JSON checks |
| `inferscope/moe.py` | Fused MoE layer in Triton, with reference and bandwidth model |
| `inferscope/bench.py` | CLI that runs every experiment and writes `results/*.json` |
| `inferscope/report.py` | Builds `REPORT.md` and plots from the results |
| `tools/extract_pulseloop_trace.py` | Rebuilds the workload trace from PulseLoop's stored runs |

## Running it

Linux or WSL2 with an NVIDIA GPU. Use a Python with development headers (for example uv-managed), because Triton compiles a small C helper at startup.

```bash
uv venv -p 3.12 && source .venv/bin/activate
uv pip install -e .

python tools/extract_pulseloop_trace.py      # needs a PulseLoop checkout
python -m inferscope.bench prep
python -m inferscope.bench hw
python -m inferscope.bench perf --variant qwen3-1.7b-bf16
python -m inferscope.report
```

Models are expected under `~/models/` (`Qwen3-1.7B`, `Qwen3-4B-AWQ`).

The raw trace is not in this repo because it contains PulseLoop's real prompts. Only its token statistics ([`results/workload.json`](results/workload.json)) are.

**No `nvcc` needed.** vLLM's FlashInfer sampler JIT-compiles CUDA at startup. `servers.py` sets `VLLM_USE_FLASHINFER_SAMPLER=0` so vLLM uses its PyTorch sampler, and attention runs on prebuilt FlashAttention.
