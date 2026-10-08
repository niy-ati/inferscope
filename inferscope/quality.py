"""Does a cheaper model variant still do the customer's job?

Three checks, all run inside one vLLM process per variant so numbers are
comparable across fp16 / FP8 / AWQ / GPTQ:

  wikitext_ppl   perplexity on WikiText-2 test (the standard quant metric)
  workload_nll   mean negative log-likelihood of the customer's *reference*
                 answers (what PulseLoop's production model actually wrote),
                 teacher-forced, given the real prompts: in-domain quality
  json           greedy generation on the real prompts, free vs. schema-guided:
                 parse rate, schema-valid rate, and agreement with the
                 baseline variant's outputs
"""

from __future__ import annotations

import json
import math
from typing import Any

import jsonschema


def _token_nll(llm: Any, token_ids: list[list[int]], spans: list[tuple[int, int]]) -> tuple[float, int]:
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt

    sp = SamplingParams(max_tokens=1, prompt_logprobs=0, temperature=0.0)
    outs = llm.generate([TokensPrompt(prompt_token_ids=t) for t in token_ids], sp, use_tqdm=False)
    total, count = 0.0, 0
    for out, ids, (lo, hi) in zip(outs, token_ids, spans):
        for pos in range(max(lo, 1), hi):
            lp = out.prompt_logprobs[pos][ids[pos]].logprob
            total -= lp
            count += 1
    return total, count


def wikitext_ppl(llm: Any, tok: Any, text: str, chunk: int = 1024, max_chunks: int = 40) -> float:
    ids = tok(text, add_special_tokens=False)["input_ids"]
    chunks = [ids[i:i + chunk] for i in range(0, len(ids) - chunk, chunk)][:max_chunks]
    nll, n = _token_nll(llm, chunks, [(0, len(c)) for c in chunks])
    return math.exp(nll / n)


def _chat(tok: Any, req: dict[str, Any], answer: str | None = None) -> str:
    msgs = [{"role": "system", "content": req["system"]}, {"role": "user", "content": req["prompt"]}]
    if answer is not None:
        msgs.append({"role": "assistant", "content": answer})
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=answer is None, enable_thinking=False)


def workload_nll(llm: Any, tok: Any, trace: list[dict[str, Any]], max_len: int) -> float:
    """Mean NLL per token of the reference answers, conditioned on the prompts."""
    ids, spans = [], []
    for req in trace:
        head = tok(_chat(tok, req), add_special_tokens=False)["input_ids"]
        full = tok(_chat(tok, req, req["output"]), add_special_tokens=False)["input_ids"]
        if len(full) > max_len:
            continue
        ids.append(full)
        spans.append((len(head), len(full)))
    nll, n = _token_nll(llm, ids, spans)
    return nll / n


def generate_json(llm: Any, tok: Any, trace: list[dict[str, Any]], guided: bool) -> list[str]:
    from vllm import SamplingParams

    params = []
    for req in trace:
        kw: dict[str, Any] = {"temperature": 0.0, "max_tokens": max(256, 2 * req["output_tokens"])}
        if guided:
            try:
                from vllm.sampling_params import GuidedDecodingParams
                kw["guided_decoding"] = GuidedDecodingParams(json=req["schema"])
            except ImportError:  # newer vLLM
                from vllm.sampling_params import StructuredOutputsParams
                kw["structured_outputs"] = StructuredOutputsParams(json=req["schema"])
        params.append(SamplingParams(**kw))
    prompts = [_chat(tok, r) for r in trace]
    return [o.outputs[0].text for o in llm.generate(prompts, params, use_tqdm=False)]


def json_scores(trace: list[dict[str, Any]], texts: list[str], baseline: list[str] | None = None) -> dict[str, float]:
    parsed = valid = same = 0
    for req, text, i in zip(trace, texts, range(len(texts))):
        s = text.strip()
        if s.startswith("```"):
            s = s.strip("`").split("\n", 1)[-1]
        try:
            obj = json.loads(s)
            parsed += 1
            jsonschema.validate(obj, req["schema"])
            valid += 1
        except (json.JSONDecodeError, jsonschema.ValidationError, jsonschema.SchemaError):
            pass
        if baseline is not None and baseline[i].strip() == text.strip():
            same += 1
    n = len(texts)
    out = {"parse_rate": parsed / n, "schema_valid_rate": valid / n}
    if baseline is not None:
        out["identical_to_baseline"] = same / n
    return out
