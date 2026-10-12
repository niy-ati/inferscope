# inferscope

How well can a real LLM workload be served on a 6 GB laptop GPU, which engine and settings serve it best, and does a first-principles model predict what the engine actually does?

The workload is [PulseLoop](https://github.com/niy-ati/PulseLoop)'s production LLM traffic: planner, copywriting, captions, judge and video-director calls. It was rebuilt from stored runs and replayed against vLLM and SGLang. Every number below comes from a file in [`results/`](results/), and every timed run records the GPU's clock, power and throttle reason alongside its latencies.

## Findings

1. **FP8 KV cache: +74% throughput, −95% p95 TTFT at 16 concurrent requests.** In the same KV memory, FP8 holds exactly twice as many tokens (14,512 vs 7,248). On Ampere it is only reachable through FlashInfer, which needed a CUDA toolchain matched to torch's headers.
2. **Prefix caching: −70% p95 TTFT at 16 concurrent requests** (2.6 s → 0.8 s), with 37% of prompt tokens served from cache, close to the 39% upper bound computed from the trace.
3. **vLLM's KV budget depends on its compile cache.** On a cold `torch.compile` cache, vLLM's memory profiler counted compiler memory as activations (0.70 vs 0.21 GiB) and cut the KV cache by 37%, making p95 TTFT **17× worse** with identical flags. The fix is to pin `--kv-cache-memory-bytes`. (The flag vLLM's own log recommends, `--kv-cache-memory`, does not exist.)
4. **vLLM 0.31 beat SGLang 0.5.9 on this workload.** At matched KV capacity it led by 10% throughput and 4× median TTFT. At each engine's best memory setting it fit 31% more KV tokens in less memory, for **+32% throughput and 15× lower p95 TTFT** at 16 concurrent requests.
5. **A 4-bit 4B model beats a bf16 1.7B model on speed and on understanding the workload:** faster decode at every concurrency up to 8, lower WikiText-2 perplexity (16.54 vs 17.77) and ~39% lower perplexity on PulseLoop's real answers. The bf16 model was more reliable at guided JSON (97% vs 89% valid).
6. **A laptop GPU on battery runs at half power.** Unplugged, the card fell from ~43 W / 1,750 MHz to ~23 W / 820 MHz (throttle reasons `0x24`). The clock sampler caught it; throttled runs are excluded from the results below.
7. **Self-hosting needs guided decoding plus a loop guard.** Without a grammar, 0% of outputs were valid JSON. With one, the remaining failures were greedy-decoding loops running to the context limit.

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

† **Unresolved:** bf16 at concurrency 1 decoded at 45.8 ms/token, 55% above its floor and slower than two requests at once. These sweeps ran before clock sampling was added. The later finding that battery power halves the GPU's clocks makes power state the leading suspect, but that is not confirmed.

### Quality

Results: [`quality_qwen3-1.7b-bf16.json`](results/quality_qwen3-1.7b-bf16.json), [`quality_qwen3-4b-awq.json`](results/quality_qwen3-4b-awq.json). 37 held-out requests. JSON generation uses greedy decoding and the same output budget as production.

| | Qwen3-1.7B bf16 | Qwen3-4B AWQ (4-bit) |
|---|---|---|
| WikiText-2 perplexity | 17.77 | **16.54** |
| NLL of PulseLoop's real answers (per token) | 3.381 | **2.886** |
| Valid JSON, unguided | 0% | 0% |
| Valid JSON, schema-guided | **97.3%** | 89.2% |
| Valid JSON, guided with no free whitespace | 89.2% | 91.9% |

- **The 4-bit 4B model understands this workload better:** its likelihood of the answers PulseLoop's production model actually wrote corresponds to about 39% lower in-domain perplexity.
- **Guided decoding is not optional on a self-hosted engine.** PulseLoop's prompts never ask for JSON; production relies on the API's strict schema mode. Without a grammar, neither model produced a single valid object.
- **The remaining failures are degenerate loops, not formatting.** Under greedy decoding a model sometimes repeats whitespace or text inside a JSON string until the context runs out (one output reached 79,638 characters). Banning whitespace *between* tokens does not stop loops *inside* strings, which is why it helped one model and hurt the other. The fix belongs in serving: a `max_tokens` close to the expected length and a retry at non-zero temperature.
- **Methodology note:** an earlier run capped output at 2× the reference length and reported 81% valid for both models. Most of those failures were that cap, which production does not have.

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

- GPTQ W4A16 with generic vs workload calibration (code in `quantize.py`, not yet run)
- Repeated runs for confidence intervals
- Distributed serving and AMD GPUs: one NVIDIA GPU was available

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
| `scripts/` | The exact benchmark queues that produced the results |
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
