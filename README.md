# inferscope

How well can a real LLM workload be served on a 6 GB laptop GPU, which engine and settings serve it best, and does a first-principles model predict what the engine actually does?

The workload is [PulseLoop](https://github.com/niy-ati/PulseLoop)'s production LLM traffic: planner, copywriting, captions, judge and video-director calls. It was rebuilt from stored runs and replayed against vLLM and SGLang. Every number below comes from a file in [`results/`](results/), and every timed run records the GPU's clock, power and throttle reason alongside its latencies.

## Findings

Speed results at concurrency 16 are means of 3 runs, each with a fresh server, on AC power at ~1,750 MHz (± is the standard deviation).

1. **FP8 KV cache: +83% throughput (240.5 ± 2.4 vs 131.7 ± 1.3 tok/s) and −96% p95 TTFT** in the same KV memory. FP8 holds exactly twice the tokens. On Ampere it is only reachable through FlashInfer, which needed a CUDA toolchain matched to torch's headers.
2. **Prefix caching: up to −70% p95 TTFT,** with 37% of prompt tokens served from cache, close to the 39% upper bound computed from the trace. How much it helps depends on KV room: with a tight KV budget, requests queue for memory and the gain shrinks to −11%.
3. **vLLM's KV budget depends on its compile cache.** On a cold `torch.compile` cache, vLLM's memory profiler counted compiler memory as activations (0.70 vs 0.21 GiB) and cut the KV cache by 37%, making p95 TTFT **17× worse** with identical flags. Pinning `--kv-cache-memory-bytes` removes it. (The flag vLLM's own log recommends, `--kv-cache-memory`, does not exist.)
4. **vLLM 0.31 beat SGLang 0.5.9 on this workload: +25% throughput (205.0 ± 1.7 vs 164.3 ± 2.5 tok/s) and 13× lower p95 TTFT,** each engine at its best memory setting. vLLM fits more KV in less memory; at matched KV capacity the two were within 10% on throughput.
5. **Tensor parallelism on 2× T4 scales decode almost linearly over PCIe** (45.4 → 22.6 ms/token for Qwen3-4B), gives 4.5× the KV capacity, and serves Qwen3-8B, which doesn't fit one T4, faster than one T4 serves Qwen3-4B.
6. **Calibrate quantization on the customer's traffic.** Two GPTQ W4A16 models differing only in calibration text: workload calibration kept guided-JSON validity at 84% vs 60% for generic text, at a small cost in WikiText perplexity. Both decode 2× faster than bf16.
7. **Loss on another model's answers is not a quality metric for quantization.** The generically calibrated GPTQ model scored a *lower* loss on PulseLoop's reference answers than its own bf16 parent while agreeing with them *less* often (top-1 62.1% vs 62.9%): its predictions are 53% higher in entropy, and spread-out probability is rewarded by that metric.
8. **Self-hosting needs guided decoding plus a loop guard.** Without a grammar, 0% of outputs were valid JSON. With one, the remaining failures were greedy-decoding loops running to the context limit.
9. **A laptop GPU on battery runs at half power** (~43 W / 1,750 MHz → ~23 W / 820 MHz). Every run records its clocks; runs below 1,600 MHz are flagged and excluded. That also explained an apparent anomaly: single-request decode measured 55% above its bandwidth floor on battery, and within 3% of it on AC.

## Hardware

NVIDIA RTX 3050 6 GB Laptop GPU (Ampere, sm_86), running under WSL2.

| | Datasheet | Measured |
|---|---|---|
| Memory bandwidth | 168 GB/s | 111 GB/s (copy), 117 GB/s (read) |
| fp16 tensor throughput | 21.5 TFLOPS | 12.2 TFLOPS |

The card sustains about 70% of its datasheet bandwidth and 57% of its compute. The performance model uses the measured ceilings, re-probed right before each run.

## Workload

74 requests from real PulseLoop runs, tokenized with the Qwen3 chat template.

| | min | p50 | p95 | max |
|---|---|---|---|---|
| Prompt tokens | 270 | 495 | 1,128 | 1,971 |
| Output tokens | 72 | 245 | 703 | 1,751 |

39% of prompt tokens sit in system prompts shared across requests, which is what makes prefix caching worth measuring. Every PulseLoop request is constrained by a JSON schema.

## Results

### KV-cache study (vLLM 0.31, Qwen3-1.7B bf16)

The whole trace in its real arrival order, with a cold prefix cache at the start of each level. All runs at ~1,740 MHz on AC power. Results: [`kv_qwen3-1.7b-bf16.json`](results/kv_qwen3-1.7b-bf16.json).

**Prefix caching** (FlashAttention backend, 12,608 KV tokens):

| | Concurrency 8: off → on | Concurrency 16: off → on |
|---|---|---|
| TTFT p50 | 245 → 194 ms | 342 → 246 ms |
| TTFT p95 | 568 → 393 ms (−31%) | 2,642 → 794 ms (**−70%**) |
| Output throughput | 153 → 164 tok/s (+7%) | 192 → 209 tok/s (+9%) |
| Prompt tokens from cache | 0 → 36.7% | 0 → 36.8% |

**FP8 KV cache** (both arms on FlashInfer, same 0.78 GiB of KV memory):

| Concurrency 16 | bf16 KV | FP8 KV |
|---|---|---|
| KV tokens | 7,248 | 14,512 (2.0×) |
| Output throughput | 139.8 tok/s | 243.1 tok/s (**+74%**) |
| TTFT p95 | 14,744 ms | 808 ms (**−95%**) |
| TPOT p50 | 41.2 ms | 39.7 ms |

At concurrency 8 the gain is +38% throughput. FP8 helps because this card is capacity-bound: with ~7k tokens, only about two worst-case requests fit at once, and the rest queue.

**What FP8 KV needs on Ampere.** FlashAttention (vLLM's default here) has no FP8 KV cache, and vLLM's Triton backend refuses it below sm_89. FlashInfer supports it but JIT-compiles kernels at startup, and its CCCL headers require an `nvcc` matching the CUDA headers torch was built with. Getting there took a pinned CUDA 13.0 toolchain (compiler, NVVM front end, `crt` headers and runtime, all the same version), a conda-forge host compiler, and `lib64` link paths. See `server_env` in [`servers.py`](inferscope/servers.py).

**The compile cache matters.** vLLM sizes the KV cache from a profiling pass. On a cold `torch.compile` cache, that pass recorded 0.70 GiB of "peak activation" instead of 0.21 GiB, because it counted the compiler's own allocations:

| Same flags, prefix caching on | Warm compile cache | Cold compile cache |
|---|---|---|
| Profiled peak activation | 0.21 GiB | 0.70 GiB |
| KV cache | 12,608 tokens | 7,984 tokens (−37%) |
| TTFT p95 at concurrency 16 | 794 ms | 13,816 ms |

Any change to the server flags invalidates the cache, so this hits exactly when a deployment is reconfigured. [`deploy/`](deploy/) pins the budget with `--kv-cache-memory-bytes`.

### vLLM vs SGLang

Same model (Qwen3-1.7B bf16), same trace in the same order, same fixed output lengths, prefix caching on. Results: [`kv_sglang_qwen3-1.7b-bf16.json`](results/kv_sglang_qwen3-1.7b-bf16.json).

| Concurrency 16 | vLLM, KV 7,984 tokens | SGLang default, KV 7,716 | SGLang tuned (`--mem-fraction-static 0.86`), KV 9,595 | vLLM, warm compile cache, KV 12,608 |
|---|---|---|---|---|
| Output throughput | 149.3 tok/s | 135.5 tok/s | 158.8 tok/s | **209.4 tok/s** |
| TTFT p50 | 711 ms | 2,852 ms | 879 ms | **246 ms** |
| TTFT p95 | 13,816 ms | 16,113 ms | 11,619 ms | **794 ms** |
| Prompt tokens from cache | 34.7% | 31.6% | 32.8% | 36.8% |
| Peak GPU memory | 4.71 GB | 5.35 GB | 5.58 GB | 5.19 GB |

- **At matched KV capacity the engines are close.** vLLM leads by 10% on throughput and 4× on median TTFT, from a smaller memory footprint.
- **Capacity decides the result, and vLLM fits more of it.** Each engine at its best memory setting: vLLM holds 12,608 KV tokens in 5.19 GB, SGLang 9,595 in 5.58 GB. That gap gives vLLM **+32% throughput and 15× lower p95 TTFT** at 16 concurrent requests.
- SGLang's RadixAttention and vLLM's block-hash prefix cache found almost the same reuse on this trace (32–37% of prompt tokens).
- SGLang 0.5.9 also needed a CUDA 12.8 toolchain matched to its torch (from conda-forge, since the CUDA 12 pip wheel ships no `nvcc`), `libnuma`, and an explicit `-ccbin` because its kernel builds ignore `CC`/`CXX`.

### Serving Qwen3-1.7B (bf16) vs Qwen3-4B (AWQ, 4-bit)

32 requests per concurrency level, closed loop. Results: [`perf_qwen3-1.7b-bf16.json`](results/perf_qwen3-1.7b-bf16.json), [`perf_qwen3-4b-awq.json`](results/perf_qwen3-4b-awq.json).

| | Qwen3-1.7B bf16 | Qwen3-4B AWQ |
|---|---|---|
| Weights | 3.22 GiB | 2.50 GiB |
| KV cache that fits | 12,608 tokens | 10,496 tokens |
| Decode floor (weights ÷ 117 GB/s) | 29.5 ms/token | 22.9 ms/token |

| Concurrency | bf16 tok/s | AWQ tok/s | bf16 TPOT | AWQ TPOT |
|---|---|---|---|---|
| 1 | 21.0 † | 38.0 | 45.8 ms † | 24.9 ms |
| 2 | 57.4 | 63.3 | 33.3 ms | 29.6 ms |
| 4 | 96.4 | 104.4 | 35.4 ms | 32.4 ms |
| 8 | 127.9 | 138.6 | 39.2 ms | 36.3 ms |
| 16 | 150.3 | 151.5 | 44.7 ms | 44.3 ms |

- **Decode is bandwidth-bound, and the model predicts it.** AWQ's single-request decode (24.9 ms) is within 9% of the weight-read floor. From 2 to 8 concurrent requests, per-token latency rises only about 20% while throughput more than doubles.
- **A 4-bit 4B model decodes faster than a bf16 1.7B one,** because it reads fewer bytes per token. At 16 concurrent requests both hit the same ceiling.
- **Prefill is where AWQ pays.** Its time to first token is higher from concurrency 4 up, since 4-bit weights are dequantized in compute-bound prefill.

† **Resolved: power state, not the engine.** This sweep measured bf16 single-request decode at 45.8 ms/token, 55% above the floor. It ran before clock sampling existed. Rerun on AC power with clocks recorded (1,752 MHz), the same request mix decodes at **30.5 ms/token, within 3% of the 29.5 ms floor** ([`perf_qwen3-1.7b-bf16-clock.json`](results/perf_qwen3-1.7b-bf16-clock.json)).

### Quality

Results: [`quality_qwen3-1.7b-bf16.json`](results/quality_qwen3-1.7b-bf16.json), [`quality_qwen3-4b-awq.json`](results/quality_qwen3-4b-awq.json). 37 held-out requests. JSON generation uses greedy decoding and the same output budget as production.

| | Qwen3-1.7B bf16 | Qwen3-4B AWQ (4-bit) |
|---|---|---|
| WikiText-2 perplexity | 17.77 | **16.54** |
| NLL of PulseLoop's reference answers (per token, see caveat) | 3.381 | 2.886 |
| Valid JSON, unguided | 0% | 0% |
| Valid JSON, schema-guided | **97.3%** | 89.2% |
| Valid JSON, guided with no free whitespace | 89.2% | 91.9% |

- **The 4-bit 4B model has the lower WikiText-2 perplexity** (16.54 vs 17.77). Its lower loss on PulseLoop's reference answers is *not* evidence of better quality: see "Loss on another model's answers" below, which shows that metric rewards less confident models.
- **Guided decoding is not optional on a self-hosted engine.** PulseLoop's prompts never ask for JSON; production relies on the API's strict schema mode. Without a grammar, neither model produced a single valid object.
- **The remaining failures are degenerate loops, not formatting.** Under greedy decoding a model sometimes repeats whitespace or text inside a JSON string until the context runs out (one output reached 79,638 characters). Banning whitespace *between* tokens does not stop loops *inside* strings, which is why it helped one model and hurt the other. The fix belongs in serving: a `max_tokens` close to the expected length and a retry at non-zero temperature.
- **Methodology note:** an earlier run capped output at 2× the reference length and reported 81% valid for both models. Most of those failures were that cap, which production does not have.

### GPTQ calibration: generic text vs the customer's own traffic

Qwen3-1.7B quantized twice with GPTQ W4A16 (llmcompressor), changing only the calibration text. Both sets carry the same number of tokens (31,352 generic vs 30,917 workload); workload calibration uses the even-id half of the trace, and quality is scored on the odd-id half. Results: [`quality_qwen3-1.7b-w4a16-*.json`](results/), [`perf_qwen3-1.7b-w4a16-*.json`](results/).

| Qwen3-1.7B | bf16 | GPTQ, generic calibration | GPTQ, workload calibration |
|---|---|---|---|
| WikiText-2 perplexity | 17.77 | **18.93** | 19.87 |
| Valid JSON, schema-guided | 97.3% | 59.5% | **83.8%** |
| Weights | 3.22 GiB | 1.27 GiB | 1.27 GiB |
| TPOT at concurrency 1 | 30.5 ms | **15.0 ms** | **15.0 ms** |
| Output throughput at concurrency 16 | 150 tok/s | 282 tok/s | 279 tok/s |

- **Each calibration set preserves what it saw.** Generic calibration keeps generic text closer to bf16; workload calibration keeps PulseLoop's structured-output reliability **24 points higher**. For a customer deployment, calibrate on the customer's traffic.
- **4-bit halves decode time,** because decode reads 1.27 instead of 3.22 GiB of weights per token, and it leaves room for 2.5× the KV cache (31,440 tokens).

**Loss on another model's answers.** Every quantized model was also scored on the loss of PulseLoop's reference answers, which a different, much larger production model wrote. The generically calibrated model scored *better* than its own bf16 parent, which no 4-bit rounding should achieve. Recomputed independently with Hugging Face transformers ([`tools/reference_nll_check.py`](tools/reference_nll_check.py), [`reference_nll_check.json`](results/reference_nll_check.json)):

| Qwen3-1.7B, 12,717 reference tokens | Loss (NLL/token) | Prediction entropy | Top-1 agreement with reference |
|---|---|---|---|
| bf16 | 3.381 | 0.361 | **62.9%** |
| GPTQ, generic calibration | **2.860** | 0.551 | 62.1% |
| GPTQ, workload calibration | 3.372 | 0.370 | 62.8% |

The generic model agrees with the reference *less* often, yet its loss is lower, because its predictions carry 53% more entropy: on text a model would not have written itself, spreading probability is rewarded. So this loss is not a quality metric for quantization. Top-1 agreement and task-level checks (schema-valid JSON) are, and both put workload calibration closest to bf16.

### Tensor parallelism on 2× T4 (Kaggle)

A single 6 GB GPU can't show distributed serving, so the same harness, trace and output lengths ran on Kaggle's free 2× Tesla T4 (Turing, 15 GB each, PCIe, no NVLink) with vLLM, fp16 and the Triton attention backend. Script: [`scripts/kaggle_t4_tp.py`](scripts/kaggle_t4_tp.py). Results: [`results/t4/`](results/t4/). Measured per T4: 276 GB/s read bandwidth, 24.7 fp16 TFLOPS.

| | Qwen3-4B, TP=1 | Qwen3-4B, TP=2 | Qwen3-8B, TP=2 |
|---|---|---|---|
| Weights per GPU | 7.56 GiB | 3.82 GiB | 7.64 GiB |
| KV cache | 25,088 tokens | **112,176 tokens** | 56,224 tokens |
| TPOT at concurrency 1 | 45.4 ms | **22.6 ms** | 37.6 ms |
| Share of the bandwidth floor | 65% | 66% | 79% |
| Output throughput, concurrency 32 | 124 tok/s | **226 tok/s** | 163 tok/s |
| TTFT p95, concurrency 32 | 466 ms | 425 ms | 534 ms |

- **Decode scales almost linearly with TP=2, even over PCIe.** Each GPU reads half the weights, and both runs reach the same ~65% of their bandwidth floor, so the all-reduce between layers costs little at this size.
- **TP=2 makes a model servable that doesn't fit one GPU, and it beats the smaller model on one GPU.** Qwen3-8B (16 GB in fp16) doesn't fit a 15 GB T4; on two it decodes faster (37.6 ms) than Qwen3-4B does on one (45.4 ms).
- **Splitting weights frees memory for KV:** 4.5× the KV capacity for the 4B model.
- **Caveat:** T4s are power-capped at 70 W, and SM clocks varied between runs (747–1,339 MHz on GPU 0). Decode is bandwidth-bound, so this matters less than compute would, but it is not as controlled as the laptop runs. One run per configuration.

### Repeated runs

Each configuration 3 times at concurrency 16, each with a fresh server, interleaved across configurations so slow drift spreads evenly. The KV budget is pinned with `--kv-cache-memory-bytes` so capacity doesn't depend on compile-cache state. Runs below 1,600 MHz are excluded (three were, when the charger was unplugged mid-run). Results: [`kv_repeats_*.json`](results/).

| Concurrency 16, mean ± sd | KV tokens | Output tok/s | TTFT p50 | TTFT p95 |
|---|---|---|---|---|
| vLLM, 0.70 GiB KV, prefix off | 6,544 | 119.5 ± 0.3 | 6,422 ± 114 ms | 24.3 ± 0.5 s |
| vLLM, 0.70 GiB KV, prefix on | 6,544 | 131.3 ± 0.9 | 2,666 ± 8 ms | 21.7 ± 0.05 s |
| vLLM, 0.70 GiB KV, FlashInfer, bf16 KV | 6,544 | 131.7 ± 1.3 | 2,633 ± 7 ms | 21.7 ± 0.2 s |
| vLLM, 0.70 GiB KV, FlashInfer, **FP8 KV** | 13,104 | **240.5 ± 2.4** | **244 ± 13 ms** | **0.95 ± 0.18 s** |
| vLLM, 1.25 GiB KV, prefix on | 11,680 | 205.0 ± 1.7 | 274 ± 10 ms | 0.86 ± 0.05 s |
| SGLang, `--mem-fraction-static 0.86`, prefix on | 9,595 | 164.3 ± 2.5 | 996 ± 56 ms | 11.2 ± 0.5 s |

Throughput varies by about 1% run to run, so every effect reported here is far outside the noise. Single runs were accurate too: the first FP8 measurement (+74%) and the repeated one (+83%) differ mainly because the bf16 arm's KV budget was 7,248 tokens then and 6,544 here.

### AMD MI300X: prediction, not yet measured

> **Status: prediction made on 2026-10-13, before any MI300X run.** Nothing in this section is a measurement. [`scripts/cloud_multi_gpu.sh`](scripts/cloud_multi_gpu.sh) will measure the same quantities on MI300X (ROCm) once AMD Developer Cloud credits are available, and this section will then report predicted vs measured.

[`tools/predict_mi300x.py`](tools/predict_mi300x.py) applies the roofline model with MI300X's datasheet figures (5.3 TB/s HBM3, 192 GB) and two efficiency bands measured on hardware that was run: sustained bandwidth at 69–86% of datasheet, and single-request decode at 65–97% of the bandwidth floor. Under tensor parallelism it adds two all-reduces per layer at an *assumed* 5–20 µs each. It predicts only single-request decode time and KV capacity; throughput under load depends on prefill interleaving and queueing, which a roofline does not model. Output: [`predictions/mi300x_qwen3-8b.json`](results/predictions/mi300x_qwen3-8b.json).

Back-test on the T4 runs: all three measured decode times fall inside the predicted bands (Qwen3-4B TP=1: 30–56 ms predicted, 45.4 measured; TP=2: 15–29, 22.6; Qwen3-8B TP=2: 29–54, 37.6). This is a consistency check, not independent validation, since the T4 runs also set the bands.

| Qwen3-8B bf16 on MI300X | TP=1 | TP=2 | TP=4 | TP=8 |
|---|---|---|---|---|
| Predicted TPOT, single request | 3.4–6.4 ms | 2.1–4.6 ms | 1.2–3.0 ms | 0.8–2.2 ms |
| Predicted KV capacity | ~1.0M tokens | ~2.2M | ~4.5M | ~9.0M |
| Worst-case trace requests that fit | 277 | 585 | 1,200 | 2,430 |

What the prediction commits to, so the measurement can prove it wrong:
- **Tensor parallelism beyond 2 pays off less for an 8B model.** At TP=8, the assumed all-reduce time is a large share of each token's decode time.
- **KV capacity stops being the constraint.** One MI300X holds hundreds of this trace's worst-case requests, where the 6 GB laptop held two or three. The bottleneck should move from memory capacity to bandwidth and scheduling.

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

### Not done

- AMD GPUs, measured: [`scripts/cloud_multi_gpu.sh`](scripts/cloud_multi_gpu.sh) runs the comparison on MI300X with ROCm and checks the prediction above; awaiting AMD Developer Cloud credits
- Multi-node serving (pipeline parallelism across machines)

## Deployment

[`deploy/`](deploy/) turns the findings into a Kubernetes deployment for vLLM. Every non-default setting cites the measurement behind it: prefix caching, FP8 KV on FlashInfer, a pinned KV budget, a persistent compile cache, a long startup probe for 2–6 minute cold starts, and KEDA autoscaling on queue depth instead of GPU utilisation. It has not been run on a cluster.

## Layout

| Path | What it does |
|---|---|
| `inferscope/hw.py` | Datasheet vs. measured bandwidth and fp16 throughput |
| `inferscope/roofline.py` | Predicts prefill time, decode step time and KV capacity from model shape and hardware ceilings |
| `inferscope/loadgen.py` | Closed-loop replay of a trace against any OpenAI-compatible server; records prefix-cache hits per request |
| `inferscope/servers.py` | Starts and stops vLLM and SGLang with matched CUDA toolchains; samples GPU memory, clocks, power and throttle reasons |
| `inferscope/quantize.py` | GPTQ W4A16 with two calibration sets |
| `inferscope/quality.py` | Perplexity, workload likelihood and JSON-schema checks |
| `inferscope/moe.py` | Fused MoE layer in Triton, with reference and bandwidth model |
| `inferscope/bench.py` | CLI that runs every experiment and writes `results/*.json` |
| `inferscope/report.py` | Builds `REPORT.md` and plots from the results |
| `tools/extract_pulseloop_trace.py` | Rebuilds the workload trace from PulseLoop's stored runs |
| `tools/predict_mi300x.py` | Roofline prediction for MI300X, back-tested on the T4 runs (prediction only) |
| `tools/reference_nll_check.py` | Loss, entropy and top-1 agreement on reference answers, with Hugging Face transformers |
| `scripts/` | The exact benchmark queues that produced the results, the Kaggle 2× T4 run, and the multi-GPU / AMD cloud script |
| `deploy/` | Kubernetes manifests derived from the results |

## Running it

Linux or WSL2 with an NVIDIA GPU. Use a Python with development headers (for example uv-managed), because Triton compiles a small C helper at startup. Plug the laptop in: on battery the results are not comparable.

```bash
uv venv -p 3.12 && source .venv/bin/activate
uv pip install -e .

python tools/extract_pulseloop_trace.py      # needs a PulseLoop checkout
python -m inferscope.bench prep
python -m inferscope.bench hw
python -m inferscope.bench perf --variant qwen3-1.7b-bf16
python -m inferscope.bench kv --variant qwen3-1.7b-bf16 --concurrency 8,16 --configs prefix_off,prefix_on
python -m inferscope.bench kv --variant qwen3-1.7b-bf16 --concurrency 8,16 --engine sglang
python -m inferscope.bench quality --variant qwen3-4b-awq
```

Models are expected under `~/models/` (`Qwen3-1.7B`, `Qwen3-4B-AWQ`). SGLang runs from its own venv (`~/inferscope/.venv-sglang`), because its pinned torch differs from vLLM's.

The raw trace and the models' outputs on it are not in this repo, because they contain or derive from PulseLoop's real prompts. Only token statistics ([`results/workload.json`](results/workload.json)) and scores are.
