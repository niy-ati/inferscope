"""Replay a workload trace against any OpenAI-compatible server (vLLM,
SGLang, llama.cpp) and record per-request latency.

Closed loop: `concurrency` workers each send the next request as soon as
their previous one finishes, so the server always holds that many requests.

Two output modes:
  fixed   generate exactly the recorded output length (ignore EOS), so every
          engine and quantization does identical work; used for performance
  natural the model stops when it likes, optionally under the request's JSON
          schema (guided decoding); used for quality and for the cost of
          structured output
"""

from __future__ import annotations

import asyncio
import json
import statistics
import time
from dataclasses import asdict, dataclass, field
from typing import Any

import httpx


@dataclass
class Result:
    id: int
    purpose: str
    prompt_tokens: int = 0
    output_tokens: int = 0
    ttft_s: float = 0.0
    e2e_s: float = 0.0
    itl_s: list[float] = field(default_factory=list)  # gaps between streamed chunks
    text: str = ""
    error: str | None = None

    @property
    def tpot_s(self) -> float:
        """Time per output token after the first."""
        return (self.e2e_s - self.ttft_s) / max(1, self.output_tokens - 1)


def _body(req: dict[str, Any], model: str, mode: str, guided: bool, engine: str) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "system", "content": req["system"]}, {"role": "user", "content": req["prompt"]}],
        "stream": True,
        "stream_options": {"include_usage": True},
        "temperature": 0.0,
    }
    if mode == "fixed":
        n = req["output_tokens"]
        body["max_tokens"] = n
        if engine == "llamacpp":
            body["n_predict"], body["ignore_eos"] = n, True
        else:
            body["min_tokens"], body["ignore_eos"] = n, True
    else:
        body["max_tokens"] = max(256, int(req["output_tokens"] * 2))
        if guided:
            body["response_format"] = {"type": "json_schema", "json_schema": {"name": "out", "schema": req["schema"]}}
    if engine in ("vllm", "sglang"):
        body["chat_template_kwargs"] = {"enable_thinking": False}  # Qwen3: answer directly
    return body


async def _one(client: httpx.AsyncClient, url: str, req: dict[str, Any], body: dict[str, Any]) -> Result:
    r = Result(id=req["id"], purpose=req["purpose"])
    t0 = time.perf_counter()
    last = t0
    parts: list[str] = []
    try:
        async with client.stream("POST", url, json=body, timeout=600) as resp:
            if resp.status_code != 200:
                r.error = f"{resp.status_code}: {(await resp.aread())[:200]!r}"
                return r
            async for line in resp.aiter_lines():
                if not line.startswith("data: ") or line == "data: [DONE]":
                    continue
                chunk = json.loads(line[6:])
                if chunk.get("usage"):
                    r.prompt_tokens = chunk["usage"].get("prompt_tokens", 0)
                    r.output_tokens = chunk["usage"].get("completion_tokens", 0)
                for ch in chunk.get("choices") or []:
                    delta = (ch.get("delta") or {}).get("content")
                    if delta:
                        now = time.perf_counter()
                        if not parts:
                            r.ttft_s = now - t0
                        else:
                            r.itl_s.append(now - last)
                        last = now
                        parts.append(delta)
    except (httpx.HTTPError, json.JSONDecodeError) as e:
        r.error = f"{type(e).__name__}: {e}"
    r.e2e_s = time.perf_counter() - t0
    r.text = "".join(parts)
    return r


async def replay(
    base_url: str,
    model: str,
    trace: list[dict[str, Any]],
    concurrency: int,
    mode: str = "fixed",
    guided: bool = False,
    engine: str = "vllm",
    repeat: int = 1,
) -> tuple[list[Result], float]:
    """Returns per-request results and the wall time of the whole replay."""
    queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
    for _ in range(repeat):
        for req in trace:
            queue.put_nowait(req)
    url = base_url.rstrip("/") + "/v1/chat/completions"
    results: list[Result] = []
    limits = httpx.Limits(max_connections=concurrency + 4, max_keepalive_connections=concurrency + 4)
    async with httpx.AsyncClient(limits=limits) as client:

        async def worker() -> None:
            while not queue.empty():
                req = queue.get_nowait()
                results.append(await _one(client, url, req, _body(req, model, mode, guided, engine)))

        t0 = time.perf_counter()
        await asyncio.gather(*(worker() for _ in range(concurrency)))
        wall = time.perf_counter() - t0
    return results, wall


def pct(xs: list[float], p: float) -> float:
    if not xs:
        return float("nan")
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(p / 100 * (len(xs) - 1))))]


def summarize(results: list[Result], wall: float) -> dict[str, Any]:
    ok = [r for r in results if not r.error]
    out_tok = sum(r.output_tokens for r in ok)
    return {
        "requests": len(results),
        "errors": len(results) - len(ok),
        "wall_s": round(wall, 2),
        "req_s": round(len(ok) / wall, 3),
        "output_tok_s": round(out_tok / wall, 1),
        "total_tok_s": round((out_tok + sum(r.prompt_tokens for r in ok)) / wall, 1),
        "ttft_p50_ms": round(1e3 * pct([r.ttft_s for r in ok], 50), 1),
        "ttft_p95_ms": round(1e3 * pct([r.ttft_s for r in ok], 95), 1),
        "tpot_p50_ms": round(1e3 * pct([r.tpot_s for r in ok], 50), 2),
        "tpot_p95_ms": round(1e3 * pct([r.tpot_s for r in ok], 95), 2),
        "e2e_p50_s": round(pct([r.e2e_s for r in ok], 50), 2),
        "e2e_p95_s": round(pct([r.e2e_s for r in ok], 95), 2),
        "mean_prompt_tokens": round(statistics.mean([r.prompt_tokens for r in ok]), 1) if ok else 0,
        "mean_output_tokens": round(statistics.mean([r.output_tokens for r in ok]), 1) if ok else 0,
    }


def dump(results: list[Result]) -> list[dict[str, Any]]:
    return [{**asdict(r), "tpot_s": r.tpot_s} for r in results]
