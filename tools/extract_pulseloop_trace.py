"""Rebuild PulseLoop's real LLM workload as a replayable trace.

PulseLoop only logs cost per call, not token counts or prompts. So we replay
every stored run through PulseLoop's own prompt builders with the model call
swapped for a recorder: that yields the exact (system, prompt, JSON schema)
it sends, and the output it actually stored for that step gives the real
output size.

Run from the PulseLoop backend directory (needs its database access):
    python extract_pulseloop_trace.py ../../inferscope/data/pulseloop_trace.jsonl

The trace holds prompt text, so it stays local (data/ is git-ignored);
only length statistics are published.
"""

from __future__ import annotations

import json
import sys
from typing import Any

from pulseloop import generators, judge, llm, planner, store, video

captured: list[dict[str, Any]] = []


def _record(system: str, prompt: str, schema: dict[str, Any], purpose: str, **_: Any) -> None:
    captured.append({"purpose": purpose, "system": system, "prompt": prompt, "schema": schema})
    return None  # callers fall back to their built-in path; we only want the request


def _take(output: Any) -> dict[str, Any] | None:
    """Attach the stored real output to the request just captured."""
    if not captured or output in (None, {}, []):
        return None
    req = captured.pop()
    req["output"] = json.dumps(output, ensure_ascii=False)
    return req


def main(out_path: str) -> None:
    llm.structured = _record  # every module calls llm.structured(...)
    trace: list[dict[str, Any]] = []
    for run in store.many("SELECT * FROM runs WHERE plan IS NOT NULL ORDER BY created_at", []):
        plan = run["plan"]
        planner.plan(run["brief"])
        if (r := _take({k: plan.get(k) for k in ("summary", "brief_type", "topic", "direction", "key_facts", "assets") if k in plan})):
            trace.append(r)
        for a in store.many("SELECT * FROM assets WHERE run_id=? AND content IS NOT NULL", [run["id"]]):
            c, kind, ch = a["content"], a["kind"], a["channel"]
            if kind not in ("post", "flyer", "video") or not a.get("components"):
                continue
            generators.generate_copy(kind, ch, a["components"], plan, [], 1)
            if (r := _take({k: c.get(k) for k in ("hook", "body", "cta", "hashtags", "headline", "subline", "scenes")})):
                trace.append(r)
            if kind != "post" and c.get("captions"):
                generators.post_kit(kind, ch, {**c, "captions": []}, plan)
                if (r := _take({"captions": c["captions"], "hashtags": c.get("post_tags") or []})):
                    trace.append(r)
            if a.get("judge") and a["judge"].get("source") == "llm":
                judge.evaluate(kind, ch, a["components"], c, plan)
                if (r := _take({k: a["judge"].get(k) for k in ("scores", "fixes")})):
                    trace.append(r)
            film = ((c.get("media") or {}).get("film"))
            if kind == "video" and film:
                video.direct(c, plan, video.seconds())
                if (r := _take({k: film.get(k) for k in ("title", "look", "shots", "captions")})):
                    trace.append(r)
            captured.clear()
    with open(out_path, "w", encoding="utf-8") as f:
        for i, r in enumerate(trace):
            f.write(json.dumps({"id": i, **r}, ensure_ascii=False) + "\n")
    by: dict[str, int] = {}
    for r in trace:
        by[r["purpose"].split(":")[0]] = by.get(r["purpose"].split(":")[0], 0) + 1
    print(f"{len(trace)} requests -> {out_path}", by)


if __name__ == "__main__":
    main(sys.argv[1])
