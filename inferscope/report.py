"""Turn results/*.json into REPORT.md + plots: what the workload is, what
the hardware can do, model vs measurement for every variant, and the
recommendation.

    python -m inferscope.report
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from . import roofline
from .bench import MODELS, RESULTS, ROOT, VARIANTS


def _load(name: str) -> Any:
    p = RESULTS / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def _ceil(hw: dict[str, Any], measured: bool) -> roofline.Ceilings:
    bw = hw["meas_read_bw_gbs"] if measured else hw["spec_bw_gbs"]  # decode streams weights: read-only
    fl = hw["meas_fp16_tflops"] if measured else hw["spec_fp16_tflops"]
    return roofline.Ceilings(bw, fl, hw["vram_gb"])


def predict(shape: roofline.ModelShape, hw: roofline.Ceilings, c: int, prompt: float, out: float) -> dict[str, float]:
    """Closed-loop steady state: each of `c` sequences prefills `prompt`
    tokens then decodes `out` tokens; decode steps run all c together."""
    ctx = prompt + out / 2
    step = roofline.decode_step_s(shape, hw, c, int(ctx))
    pre = roofline.prefill_s(shape, hw, int(prompt))
    # per request: its own prefill (others' prefills also stall the batch: c * pre over the request's life, shared)
    e2e = pre * c + out * step["step_s"]
    return {"tpot_ms": 1e3 * step["step_s"], "ttft_c1_ms": 1e3 * pre, "out_tok_s": c * out / e2e, "bound": step["bound"]}


def _fmt_err(pred: float, meas: float) -> str:
    return f"{(meas - pred) / pred * 100:+.0f}%" if pred and meas == meas else "n/a"


def main() -> None:
    hw, wl = _load("hw.json"), _load("workload.json")
    lines = ["# inferscope report: PulseLoop on an RTX 3050 6GB Laptop GPU", ""]
    lines += ["## 1. The customer workload", "",
              f"{wl['requests']} real PulseLoop LLM requests ({', '.join(f'{k} {v}' for k, v in wl['by_purpose'].items())}), every one constrained by a JSON schema.", "",
              "| | min | p50 | p95 | max |", "|---|---|---|---|---|",
              f"| prompt tokens | {wl['prompt_tokens']['min']} | {wl['prompt_tokens']['p50']} | {wl['prompt_tokens']['p95']} | {wl['prompt_tokens']['max']} |",
              f"| output tokens | {wl['output_tokens']['min']} | {wl['output_tokens']['p50']} | {wl['output_tokens']['p95']} | {wl['output_tokens']['max']} |", "",
              f"{wl['shared_prefix_token_share']:.0%} of prompt tokens repeat a prefix seen in an earlier request (upper bound for prefix caching).", ""]
    lines += ["## 2. Hardware ceilings: datasheet vs measured", "",
              "| | datasheet | measured | measured / datasheet |", "|---|---|---|---|",
              f"| memory bandwidth, copy (GB/s) | {hw['spec_bw_gbs']} | {hw['meas_bw_gbs']} | {hw['meas_bw_gbs'] / hw['spec_bw_gbs']:.0%} |",
              f"| memory bandwidth, read-only (GB/s) | {hw['spec_bw_gbs']} | {hw['meas_read_bw_gbs']} | {hw['meas_read_bw_gbs'] / hw['spec_bw_gbs']:.0%} |",
              f"| fp16 tensor, fp32 acc (TFLOPS) | {hw['spec_fp16_tflops']} | {hw['meas_fp16_tflops']} | {hw['meas_fp16_tflops'] / hw['spec_fp16_tflops']:.0%} |", "",
              f"Max SM clock {hw['sm_clock_mhz']} MHz, power limit {hw['power_limit_w']} W.", ""]

    lines += ["## 3. Performance model vs measurement", "",
              "Prediction uses ceilings measured right before each run (read bandwidth, fp16 TFLOPS). `err` = (measured - predicted) / predicted.", ""]
    plot: dict[str, Any] = {}
    for name, v in VARIANTS.items():
        perf = _load(f"perf_{name}.json")
        if not perf or not (MODELS / v["model"]).exists():
            continue
        shape = roofline.load_shape(MODELS / v["model"], name)
        if v.get("quantization") == "fp8":
            shape.weight_bytes /= 2
        ce = _ceil(perf.get("hw_at_run") or hw, True)
        h = perf.get("hw_at_run") or hw
        lines += [f"### {name}", "", f"Ceilings at run time: {h['meas_read_bw_gbs']} GB/s read, {h['meas_fp16_tflops']} TFLOPS.", "",
                  f"Weights read per decode step: {shape.decode_weight_bytes(1) / 1e9:.2f} GB · KV cache {shape.kv_bytes_per_token / 1024:.0f} KiB/token · "
                  f"decode turns compute-bound at batch ≈ {roofline.crossover_batch(shape, ce, int(wl['prompt_tokens']['p50']))}", "",
                  "| concurrency | TPOT pred (ms) | TPOT meas p50 | err | out tok/s pred | out tok/s meas | err | TTFT p50 (ms) | GPU peak (MiB) |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for c, s in sorted(perf["levels"].items(), key=lambda kv: int(kv[0])):
            p = predict(shape, ce, int(c), s["mean_prompt_tokens"], s["mean_output_tokens"])
            lines.append(f"| {c} | {p['tpot_ms']:.1f} | {s['tpot_p50_ms']:.1f} | {_fmt_err(p['tpot_ms'], s['tpot_p50_ms'])} | {p['out_tok_s']:.0f} | {s['output_tok_s']:.0f} | "
                         f"{_fmt_err(p['out_tok_s'], s['output_tok_s'])} | {s['ttft_p50_ms']:.0f} | {s['gpu_peak_mib']} |")
            plot.setdefault(name, []).append((int(c), p["out_tok_s"], s["output_tok_s"], p["tpot_ms"], s["tpot_p50_ms"]))
        info = perf.get("server", {})
        if info.get("kv_cache_gib") is not None:
            pred_tokens = int(info["kv_cache_gib"] * 2**30 // shape.kv_bytes_per_token)
            lines += ["", f"KV cache: vLLM reserved {info['kv_cache_gib']} GiB → {info.get('kv_cache_tokens', 'n/a'):,} tokens; "
                      f"the model predicts {pred_tokens:,} tokens for that memory ({_fmt_err(pred_tokens, info.get('kv_cache_tokens', float('nan')))}). "
                      f"Weights in memory {info.get('weights_gib', 'n/a')} GiB (checkpoint {shape.weight_bytes / 2**30:.2f} GiB)."]
        lines.append("")

    if (pf := _load("prefix_qwen3-1.7b-bf16.json")):
        on, off = pf["on"], pf["off"]
        lines += ["## 4. Prefix caching on the real request order", "",
                  "| | TTFT p50 (ms) | TTFT p95 (ms) | e2e p50 (s) | out tok/s |", "|---|---|---|---|---|",
                  f"| off | {off['ttft_p50_ms']} | {off['ttft_p95_ms']} | {off['e2e_p50_s']} | {off['output_tok_s']} |",
                  f"| on | {on['ttft_p50_ms']} | {on['ttft_p95_ms']} | {on['e2e_p50_s']} | {on['output_tok_s']} |",
                  f"| change | {_fmt_err(off['ttft_p50_ms'], on['ttft_p50_ms'])} | {_fmt_err(off['ttft_p95_ms'], on['ttft_p95_ms'])} | {_fmt_err(off['e2e_p50_s'], on['e2e_p50_s'])} | {_fmt_err(off['output_tok_s'], on['output_tok_s'])} |", ""]

    if (gd := _load("guided_qwen3-1.7b-bf16.json")):
        lines += ["## 5. Structured output (JSON-schema guided decoding)", "",
                  "| | schema-valid | parses as JSON | TPOT p50 (ms) | e2e p50 (s) | out tok/s |", "|---|---|---|---|---|---|"]
        for k in ("free", "guided"):
            g = gd[k]
            lines.append(f"| {k} | {g['schema_valid_rate']:.0%} | {g['parse_rate']:.0%} | {g['tpot_p50_ms']} | {g['e2e_p50_s']} | {g['output_tok_s']} |")
        lines.append("")

    rows = [(n, _load(f"quality_{n}.json")) for n in VARIANTS]
    rows = [(n, q) for n, q in rows if q]
    if rows:
        lines += ["## 6. Quality per variant (held-out half of the workload)", "",
                  "| variant | WikiText-2 PPL | workload NLL/token | schema-valid (free) | schema-valid (guided) | identical to bf16 (guided) |", "|---|---|---|---|---|---|"]
        for n, q in rows:
            lines.append(f"| {n} | {q['wikitext_ppl']} | {q['workload_nll']} | {q['json_free']['schema_valid_rate']:.0%} | {q['json_guided']['schema_valid_rate']:.0%} | "
                         f"{q['json_guided'].get('identical_to_baseline', 1.0):.0%} |")
        lines.append("")

    if (moe := _load("moe.json")):
        lines += ["## 7. Fused MoE layer (Triton) vs per-expert PyTorch loop", "",
                  "One MoE layer at real model shapes, fp16. `bound` = time to stream every touched expert's weights once at the measured read bandwidth.", "",
                  "| model | tokens | experts touched | fused (ms) | loop (ms) | speedup | achieved GB/s | % of read BW | max rel. error |", "|---|---|---|---|---|---|---|---|---|"]
        for r in moe:
            lines.append(f"| {r['model']} | {r['tokens']} | {r['experts_touched']} | {r['fused_ms']} | {r['naive_ms']} | {r['speedup']}x | {r['fused_gbs']} | "
                         f"{r['fused_gbs'] / hw['meas_read_bw_gbs']:.0%} | {r['max_rel_err']:.1e} |")
        lines.append("")

    if plot:
        _plots(plot)
        lines += ["![throughput](results/throughput.png)", "", "![tpot](results/tpot.png)", ""]
    (ROOT / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print("wrote", ROOT / "REPORT.md")


def _plots(plot: dict[str, Any]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for idx, fname, ylabel in ((1, "throughput.png", "output tokens / s"), (3, "tpot.png", "time per output token (ms)")):
        fig, ax = plt.subplots(figsize=(7, 4.2))
        for i, (name, pts) in enumerate(plot.items()):
            pts.sort()
            col = f"C{i}"
            ax.plot([p[0] for p in pts], [p[idx + 1] for p in pts], "o-", color=col, label=f"{name} measured")
            ax.plot([p[0] for p in pts], [p[idx] for p in pts], "--", color=col, alpha=0.6, label=f"{name} model")
        ax.set_xscale("log", base=2)
        ax.set_xlabel("concurrent requests")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(RESULTS / fname, dpi=140)
        plt.close(fig)


if __name__ == "__main__":
    main()
