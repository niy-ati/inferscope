# Deploying what the benchmarks found

[`k8s/vllm.yaml`](k8s/vllm.yaml) serves Qwen3-1.7B with vLLM. Every non-default setting comes from a measurement in [`results/`](../results/). It was written from single-GPU benchmarks and has not been run on a cluster yet.

| Setting | Why | Evidence |
|---|---|---|
| `--enable-prefix-caching` | 37% of prompt tokens hit the cache on the real trace; p95 TTFT at concurrency 16 drops from 2.6 s to 0.8 s | `kv_qwen3-1.7b-bf16.json`: `prefix_off` vs `prefix_on` |
| `--kv-cache-dtype=fp8` | Twice the KV tokens in the same memory; +74% throughput and −95% p95 TTFT at concurrency 16 | `flashinfer_bf16_kv` vs `flashinfer_fp8_kv` |
| `--attention-backend=FLASHINFER` | The only backend with an FP8 KV cache on Ampere: FlashAttention has none, and vLLM's Triton backend needs sm_89+ | server logs |
| `--kv-cache-memory-bytes` | On a cold `torch.compile` cache, vLLM's profiler counted compile memory as activations (0.70 vs 0.21 GiB) and shrank the KV cache by 37%. Pinning the budget removes the dependence on cache state. | `Actual usage is …` lines in the logs |
| Persistent cache volume | Keeps the compile and FlashInfer JIT caches across restarts: faster cold starts (2–6 min measured) | `startup_s` in results |
| Long `startupProbe` | Cold starts take 2–6 min; a default liveness probe would kill the pod while it compiles | `startup_s` in results |
| KEDA on `vllm:num_requests_waiting` | GPU utilisation sits near 90% from concurrency 2 upward, so it cannot tell healthy from overloaded; the waiting queue can | `gpu_util_mean` in results |

FlashInfer JIT-compiles its attention kernels at startup, so the image needs an `nvcc` that matches the CUDA headers torch was built with. Getting that wrong cost the most time here: a mismatched compiler, front end, header set or link path each failed differently (see `inferscope/servers.py`, `server_env`). Check this before relying on the FP8 setting in a new image.

`--kv-cache-memory-bytes=1220194386` is the budget vLLM computed with a warm cache at `--gpu-memory-utilization=0.82` on a 6 GB GPU. Recompute it for other GPUs.
