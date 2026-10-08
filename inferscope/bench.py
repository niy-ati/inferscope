"""inferscope CLI.

    python -m inferscope.bench prep    # token counts for the trace
    python -m inferscope.bench hw      # spec vs measured ceilings
    python -m inferscope.bench perf  --variant qwen3-1.7b-bf16 [--concurrency 1,2,4,8,16]
    python -m inferscope.bench prefix --variant qwen3-1.7b-bf16     # prefix cache on/off
    python -m inferscope.bench guided --variant qwen3-1.7b-bf16     # structured output cost
    python -m inferscope.bench quality --variant qwen3-1.7b-bf16
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
from pathlib import Path
from typing import Any

from . import loadgen, servers

ROOT = Path(__file__).resolve().parent.parent
DATA, RESULTS = ROOT / "data", ROOT / "results"
MODELS = Path.home() / "models"

VARIANTS: dict[str, dict[str, Any]] = {
    "qwen3-1.7b-bf16": {"model": "Qwen3-1.7B"},
    "qwen3-1.7b-fp8": {"model": "Qwen3-1.7B", "quantization": "fp8"},  # weight-only FP8 (Marlin) on Ampere
    "qwen3-1.7b-w4a16-generic": {"model": "Qwen3-1.7B-W4A16-generic"},
    "qwen3-1.7b-w4a16-workload": {"model": "Qwen3-1.7B-W4A16-workload"},
    "qwen3-4b-awq": {"model": "Qwen3-4B-AWQ"},
}
PORT = 8011


def load_trace() -> list[dict[str, Any]]:
    path = DATA / "trace.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def calib_split(trace: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Requests GPTQ may calibrate on (even ids)."""
    return [r for r in trace if r["id"] % 2 == 0]


def eval_split(trace: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Held-out requests every variant's quality is scored on (odd ids)."""
    return [r for r in trace if r["id"] % 2 == 1]


def _wikitext() -> None:
    import pyarrow.parquet as pq

    for split in ("test", "train"):
        out = DATA / f"wikitext2_{split}.txt"
        if not out.exists():
            out.write_text("".join(pq.read_table(DATA / f"wt_{split}.parquet")["text"].to_pylist()), encoding="utf-8")


def prep() -> None:
    """Count tokens with the (shared) Qwen3 tokenizer; keep the trace slim."""
    from transformers import AutoTokenizer

    _wikitext()
    tok = AutoTokenizer.from_pretrained(MODELS / "Qwen3-1.7B")
    rows = [json.loads(line) for line in (DATA / "pulseloop_trace.jsonl").read_text(encoding="utf-8").splitlines()]
    with open(DATA / "trace.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            r["prompt_tokens"] = len(chat_ids(tok, r))
            r["output_tokens"] = len(tok(r["output"], add_special_tokens=False)["input_ids"])
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    stats = {k: sorted(r[k] for r in rows) for k in ("prompt_tokens", "output_tokens")}
    shape = {k: {"min": v[0], "p50": v[len(v) // 2], "p95": v[int(0.95 * (len(v) - 1))], "max": v[-1]} for k, v in stats.items()}
    purposes: dict[str, int] = {}
    for r in rows:
        purposes[r["purpose"].split(":")[0]] = purposes.get(r["purpose"].split(":")[0], 0) + 1
    shared = _shared_prefix_share(rows, tok)
    _save("workload.json", {"requests": len(rows), "by_purpose": purposes, **shape, "shared_prefix_token_share": shared})
    print(json.dumps(shape), purposes, f"shared prefix share {shared:.0%}")


def _shared_prefix_share(rows: list[dict[str, Any]], tok: Any) -> float:
    """Share of prompt tokens that repeat a prefix already seen in an earlier
    request (what an ideal prefix cache could skip)."""
    seen: list[list[int]] = []
    hit = total = 0
    for r in rows:
        ids = chat_ids(tok, r)
        best = 0
        for s in seen:
            n = 0
            for a, b in zip(ids, s):
                if a != b:
                    break
                n += 1
            best = max(best, n)
        hit += best
        total += len(ids)
        seen.append(ids)
    return hit / total


def chat_ids(tok: Any, r: dict[str, Any]) -> list[int]:
    """Prompt token ids exactly as the server builds them (Qwen3, no thinking)."""
    msgs = [{"role": "system", "content": r["system"]}, {"role": "user", "content": r["prompt"]}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    return tok(text, add_special_tokens=False)["input_ids"]


def probe_hw() -> dict[str, Any]:
    """Laptop clocks drift with heat and power state, so every run measures
    the ceilings right before it starts. In a child process: a CUDA context
    left in this process would take GPU memory away from the server."""
    import subprocess
    import sys
    code = "import json; from dataclasses import asdict; from inferscope import hw; print(json.dumps(asdict(hw.probe())))"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=ROOT, check=True).stdout
    return json.loads(out.strip().splitlines()[-1])


def _save(name: str, obj: Any) -> None:
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / name).write_text(json.dumps(obj, indent=2), encoding="utf-8")


def _server(variant: str, **kw: Any) -> servers.Server:
    v = VARIANTS[variant]
    kw.setdefault("max_len", 4096)  # longest request: 1971 prompt + 1751 output tokens
    return servers.vllm(str(MODELS / v["model"]), PORT, RESULTS / "logs" / f"{variant}{kw.pop('tag', '')}.log", quantization=v.get("quantization"), **kw)


def _sample(trace: list[dict[str, Any]], n: int, seed: int = 0) -> list[dict[str, Any]]:
    rnd = random.Random(seed)
    return trace if n >= len(trace) else rnd.sample(trace, n)


def _replay(srv: servers.Server, reqs: list[dict[str, Any]], c: int, **kw: Any) -> dict[str, Any]:
    with servers.GpuSampler() as gpu:
        res, wall = asyncio.run(loadgen.replay(srv.url, srv.model_id(), reqs, c, **kw))
    return {**loadgen.summarize(res, wall), "gpu_peak_mib": gpu.peak_mib, "gpu_util_mean": round(gpu.mean_util, 1)}, res


def perf(variant: str, levels: list[int], n: int) -> None:
    trace = load_trace()
    reqs = _sample(trace, n)
    out: dict[str, Any] = {"variant": variant, "levels": {}, "hw_at_run": probe_hw()}
    with _server(variant) as srv:
        out["server"] = {"startup_s": round(srv.started_s, 1), **srv.info}
        _replay(srv, reqs[:4], 2)  # warm-up (CUDA graphs, allocator)
        for c in levels:
            summary, res = _replay(srv, reqs * max(1, (2 * c + len(reqs) - 1) // len(reqs)), c)
            out["levels"][c] = summary
            print(variant, f"c={c}", json.dumps(summary))
    _save(f"perf_{variant}.json", out)


def prefix(variant: str, c: int, n: int) -> None:
    """Same requests, prefix caching on vs off. Requests arrive in the
    trace's real order (shared system prompts recur), at concurrency c."""
    reqs = load_trace()[:n]
    out: dict[str, Any] = {"variant": variant, "concurrency": c}
    for on in (False, True):
        with _server(variant, prefix_cache=on, tag=f"-prefix{int(on)}") as srv:
            _replay(srv, [{**r, "system": "warm-up " + r["system"]} for r in reqs[:2]], 1)  # warm kernels, not the cache
            summary, _ = _replay(srv, reqs, c)
            out["on" if on else "off"] = summary
            print("prefix", on, json.dumps(summary))
    _save(f"prefix_{variant}.json", out)


def guided(variant: str, c: int, n: int) -> None:
    """Natural generation with vs without JSON-schema guided decoding."""
    reqs = _sample(load_trace(), n)
    out: dict[str, Any] = {"variant": variant, "concurrency": c}
    from .quality import json_scores
    with _server(variant) as srv:
        _replay(srv, reqs[:2], 1, mode="natural")
        for g in (False, True):
            summary, res = _replay(srv, reqs, c, mode="natural", guided=g)
            order = {r.id: r.text for r in res}
            out["guided" if g else "free"] = {**summary, **json_scores(reqs, [order.get(r["id"], "") for r in reqs])}
            print("guided", g, json.dumps(out["guided" if g else "free"]))
    _save(f"guided_{variant}.json", out)


def quality(variant: str, wikitext: str, baseline: str | None) -> None:
    from transformers import AutoTokenizer
    from vllm import LLM

    from . import quality as q

    v = VARIANTS[variant]
    path = str(MODELS / v["model"])
    tok = AutoTokenizer.from_pretrained(MODELS / "Qwen3-1.7B")
    llm = LLM(path, quantization=v.get("quantization"), max_model_len=4096, gpu_memory_utilization=0.85, enable_prefix_caching=False)
    trace = eval_split(load_trace())
    out: dict[str, Any] = {"variant": variant, "eval_requests": len(trace)}
    out["wikitext_ppl"] = round(q.wikitext_ppl(llm, tok, Path(wikitext).read_text(encoding="utf-8")), 3)
    out["workload_nll"] = round(q.workload_nll(llm, tok, trace, 4096), 4)
    base_free = base_guided = None
    if baseline and (RESULTS / f"gen_{baseline}.json").exists():
        b = json.loads((RESULTS / f"gen_{baseline}.json").read_text(encoding="utf-8"))
        base_free, base_guided = b["free"], b["guided"]
    free = q.generate_json(llm, tok, trace, guided=False)
    gd = q.generate_json(llm, tok, trace, guided=True)
    _save(f"gen_{variant}.json", {"free": free, "guided": gd})
    out["json_free"] = q.json_scores(trace, free, base_free)
    out["json_guided"] = q.json_scores(trace, gd, base_guided)
    print(json.dumps(out))
    _save(f"quality_{variant}.json", out)


def hw() -> None:
    _save("hw.json", probe_hw())
    print((RESULTS / "hw.json").read_text())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["prep", "hw", "perf", "prefix", "guided", "quality"])
    ap.add_argument("--variant", default="qwen3-1.7b-bf16")
    ap.add_argument("--concurrency", default="1,2,4,8,16")
    ap.add_argument("-c", type=int, default=4)
    ap.add_argument("-n", type=int, default=32)
    ap.add_argument("--wikitext", default=str(DATA / "wikitext2_test.txt"))
    ap.add_argument("--baseline", default="qwen3-1.7b-bf16")
    a = ap.parse_args()
    if a.cmd == "prep":
        prep()
    elif a.cmd == "hw":
        hw()
    elif a.cmd == "perf":
        perf(a.variant, [int(x) for x in a.concurrency.split(",")], a.n)
    elif a.cmd == "prefix":
        prefix(a.variant, a.c, a.n)
    elif a.cmd == "guided":
        guided(a.variant, a.c, a.n)
    else:
        quality(a.variant, a.wikitext, a.baseline if a.baseline != a.variant else None)


if __name__ == "__main__":
    main()
