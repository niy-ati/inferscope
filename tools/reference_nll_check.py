"""Why does a 4-bit model score a *lower* loss than its own bf16 parent on
PulseLoop's reference answers?

The reference answers were written by PulseLoop's production model, a
different and much larger LLM. Loss on another model's text rewards a model
that spreads probability more widely. This recomputes the loss with Hugging
Face transformers (independently of vLLM) and adds two numbers that separate
"agrees with the reference" from "is less confident":

  entropy       mean entropy of the next-token distribution (nats)
  top1_agree    share of reference tokens that are the model's first choice

    python tools/reference_nll_check.py Qwen3-1.7B Qwen3-1.7B-W4A16-generic Qwen3-1.7B-W4A16-workload
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from inferscope.bench import RESULTS, eval_split, load_trace  # noqa: E402
from inferscope.quality import _chat  # noqa: E402

MODELS = Path.home() / "models"


def score(name: str, tok, reqs: list[dict]) -> dict:
    model = AutoModelForCausalLM.from_pretrained(MODELS / name, torch_dtype=torch.bfloat16, device_map="cuda")
    nll = ent = top1 = 0.0
    n = 0
    for r in reqs:
        head = len(tok(_chat(tok, r), add_special_tokens=False)["input_ids"])
        ids = torch.tensor([tok(_chat(tok, r, r["output"]), add_special_tokens=False)["input_ids"]], device="cuda")
        if ids.shape[1] > 4096:
            continue
        with torch.no_grad():
            logp = torch.log_softmax(model(ids).logits[0, head - 1:-1].float(), -1)
        tgt = ids[0, head:]
        nll -= logp.gather(1, tgt[:, None]).sum().item()
        ent -= (logp.exp() * logp).sum().item()
        top1 += (logp.argmax(-1) == tgt).sum().item()
        n += tgt.numel()
    del model
    torch.cuda.empty_cache()
    return {"model": name, "tokens": n, "nll": round(nll / n, 4), "entropy": round(ent / n, 4), "top1_agree": round(top1 / n, 4)}


def main() -> None:
    tok = AutoTokenizer.from_pretrained(MODELS / "Qwen3-1.7B")
    reqs = eval_split(load_trace())
    rows = []
    for name in sys.argv[1:]:
        try:
            rows.append(score(name, tok, reqs))
        except Exception as e:  # e.g. a quantization format plain transformers can't load
            print(json.dumps({"model": name, "error": f"{type(e).__name__}: {e}"[:200]}), flush=True)
            continue
        print(json.dumps(rows[-1]), flush=True)
        (RESULTS / "reference_nll_check.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
