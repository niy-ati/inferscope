"""GPTQ W4A16 quantization with two calibration sets.

GPTQ picks each layer's rounding to minimise error on calibration
activations, so *which* text it sees matters. We quantize the same model
twice and compare:

  generic   WikiText-2 train paragraphs (the usual default)
  workload  the customer's own requests: PulseLoop prompts + reference answers
            (calibration half of the trace only; quality is scored on the
            other half, so there is no leakage)

    python -m inferscope.quantize --calib generic
    python -m inferscope.quantize --calib workload
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .bench import DATA, MODELS, calib_split, load_trace


def calibration_texts(kind: str, tok, n: int) -> list[str]:
    if kind == "workload":
        out = []
        for r in calib_split(load_trace()):
            msgs = [{"role": "system", "content": r["system"]}, {"role": "user", "content": r["prompt"]}, {"role": "assistant", "content": r["output"]}]
            out.append(tok.apply_chat_template(msgs, tokenize=False, enable_thinking=False))
        return out[:n]
    paras = [p for p in (DATA / "wikitext2_train.txt").read_text(encoding="utf-8").split("\n \n") if len(p) > 1500]
    return paras[:n]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--calib", choices=["generic", "workload"], required=True)
    ap.add_argument("-n", type=int, default=64)
    ap.add_argument("--max-len", type=int, default=2048)
    a = ap.parse_args()

    from datasets import Dataset
    from llmcompressor import oneshot
    from llmcompressor.modifiers.quantization import GPTQModifier
    from transformers import AutoModelForCausalLM, AutoTokenizer

    src = MODELS / "Qwen3-1.7B"
    dst = MODELS / f"Qwen3-1.7B-W4A16-{a.calib}"
    tok = AutoTokenizer.from_pretrained(src)
    model = AutoModelForCausalLM.from_pretrained(src, torch_dtype="auto")
    texts = calibration_texts(a.calib, tok, a.n)
    ds = Dataset.from_dict(tok(texts, truncation=True, max_length=a.max_len, add_special_tokens=False))
    recipe = GPTQModifier(targets="Linear", scheme="W4A16", ignore=["lm_head"], dampening_frac=0.01)
    oneshot(model=model, dataset=ds, recipe=recipe, max_seq_length=a.max_len, num_calibration_samples=len(texts))
    model.save_pretrained(dst, save_compressed=True)
    tok.save_pretrained(dst)
    (dst / "inferscope_calib.json").write_text(json.dumps({"calib": a.calib, "samples": len(texts), "max_len": a.max_len}))
    print("saved", dst, "with", len(texts), "calibration samples")


if __name__ == "__main__":
    main()
