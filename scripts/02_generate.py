"""Generate target-model responses for every corpus prompt.

Runs at temperature 0 so the frozen corpus is deterministic and reproducible: the label
attached to a prompt is a property of that prompt, not of a sampling draw.

Refusals are RETAINED. The 2x2 of route x outcome is the point: keeping only violations
would collapse the design back to a text-separable route classifier.
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import azclient  # noqa: E402
import corpus  # noqa: E402

TARGET_DEPLOYMENT = "phi4-target"
OUT = ROOT / "data" / "interim" / "generations.jsonl"


async def main() -> None:
    items = corpus.load(ROOT / "data" / "interim" / "corpus.jsonl")
    print(f"[gen] {len(items)} prompts")

    jobs = [
        azclient.Job(
            rid=it.rid,
            deployment=TARGET_DEPLOYMENT,
            messages=[{"role": "user", "content": it.prompt}],
            max_tokens=600,
            temperature=0.0,
            meta={"variant": it.variant, "route_family": it.route_family, "category": it.category},
        )
        for it in items
    ]

    await azclient.run_jobs(jobs, OUT, concurrency=32)

    rows = azclient.read_jsonl(OUT)
    ok = sum(1 for r in rows if r.get("ok"))
    print(f"[gen] done: {ok}/{len(rows)} succeeded")
    if ok < len(rows):
        from collections import Counter
        errs = Counter(r.get("error", "?") for r in rows if not r.get("ok"))
        for e, n in errs.most_common(10):
            print(f"  {n:4d}  {e}")


if __name__ == "__main__":
    asyncio.run(main())
