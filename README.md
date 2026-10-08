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

### Serving Qwen3-1.7B (bf16) on vLLM 0.31

32 requests per concurrency level, closed loop.

| Concurrency | Output tok/s | TTFT p50 / p95 | TPOT p50 | E2E p95 |
|---|---|---|---|---|
| 2 | 57.1 | 197 / 335 ms | 33.7 ms | 16.1 s |
| 4 | 90.4 | 228 / 489 ms | 37.4 ms | 18.2 s |
| 8 | 123.1 | 236 / 946 ms | 41.2 ms | 20.3 s |
| 16 | 132.3 | 1,919 / 10,347 ms | 43.8 ms | 30.8 s |

- **Decode is bandwidth-bound.** From 2 to 8 concurrent requests, per-token latency rises only 22% while throughput more than doubles. Each extra sequence rides on the same weight read.
- **The KV cache is the ceiling.** After weights and CUDA graphs, about 8,000 tokens of KV cache fit, which is roughly two worst-case requests. At 16 concurrent requests, throughput gains only 7% while median time-to-first-token rises 8×, because requests queue for cache space. On this card, 8 is the sensible limit.
- The concurrency-1 run is being re-measured. Its total time doesn't match its own per-token latency, which points to warm-up compilation, not serving cost.

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

- Qwen3-4B-AWQ and FP8 / GPTQ W4A16 variants of Qwen3-1.7B on the same sweep
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
