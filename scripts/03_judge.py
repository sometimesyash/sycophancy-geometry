"""Label every generation with the judge, then freeze the corpus.

Content-filter blocks from generation are labelled deterministically as
complied=False, violation=False: the platform refused on the model's behalf, which is a
refusal outcome, and it is recorded as `blocked` so it can be excluded or analysed
separately rather than silently treated as a model refusal.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import azclient  # noqa: E402
import corpus  # noqa: E402
import judge  # noqa: E402

JUDGE_DEPLOYMENT = "gpt-5-4"
INTERIM = ROOT / "data" / "interim"
FROZEN = ROOT / "data" / "frozen"
OUT = INTERIM / "judgements.jsonl"


async def main() -> None:
    items = {it.rid: it for it in corpus.load(INTERIM / "corpus.jsonl")}
    gens = {g["rid"]: g for g in azclient.read_jsonl(INTERIM / "generations.jsonl")}
    print(f"[judge] {len(items)} prompts | {len(gens)} generations")

    jobs = []
    for rid, it in items.items():
        g = gens.get(rid)
        if not g or not g.get("ok"):
            continue
        jobs.append(
            azclient.Job(
                rid=rid,
                deployment=JUDGE_DEPLOYMENT,
                messages=judge.build_messages(it.prompt, g.get("content", "")),
                max_tokens=2000,
                temperature=None,
                meta={"variant": it.variant},
            )
        )

    print(f"[judge] judging {len(jobs)} responses")
    await azclient.run_jobs(jobs, OUT, concurrency=24)

    # ---- assemble the frozen table -------------------------------------------------
    judged = {j["rid"]: j for j in azclient.read_jsonl(OUT)}
    rows = []
    for rid, it in items.items():
        g = gens.get(rid)
        rec = {
            "rid": rid,
            "case_id": it.case_id,
            "base_id": it.base_id,
            "variant": it.variant,
            "route_family": it.route_family,
            "lever_name": it.lever_name,
            "category": it.category,
            "harmful_base": it.harmful,
            "prompt": it.prompt,
        }

        if not g or not g.get("ok"):
            rec.update(
                response="", blocked=True, complied=False, violation=False,
                judge_conf=1.0, judge_ok=True, judge_rationale="platform content filter",
                heuristic_refusal=True, finish_reason="blocked",
            )
            rows.append(rec)
            continue

        resp = g.get("content", "")
        jj = judged.get(rid)
        parsed = judge.parse(jj.get("content", "")) if jj and jj.get("ok") else {"parse_ok": False, "reason": "no_judge"}

        rec.update(
            response=resp,
            blocked=False,
            finish_reason=g.get("finish_reason"),
            heuristic_refusal=judge.looks_like_refusal(resp),
            judge_ok=bool(parsed.get("parse_ok")),
            complied=parsed.get("complied"),
            violation=parsed.get("violation"),
            judge_conf=parsed.get("confidence"),
            judge_rationale=parsed.get("rationale", parsed.get("reason", "")),
        )
        rows.append(rec)

    df = pd.DataFrame(rows)
    FROZEN.mkdir(parents=True, exist_ok=True)
    out_pq = FROZEN / "corpus_labelled.parquet"
    out_csv = FROZEN / "corpus_labelled.csv"
    df.to_parquet(out_pq, index=False)
    df.to_csv(out_csv, index=False)

    digest = hashlib.sha256(out_pq.read_bytes()).hexdigest()

    # ---- report --------------------------------------------------------------------
    print(f"\n[judge] rows={len(df)}  judge_ok={int(df.judge_ok.sum())}  blocked={int(df.blocked.sum())}")
    print("\n=== VIOLATION RATE BY VARIANT ===")
    print(df.groupby("variant")["violation"].agg(["mean", "sum", "count"]).round(3))
    print("\n=== COMPLIANCE RATE BY VARIANT ===")
    print(df.groupby("variant")["complied"].agg(["mean", "count"]).round(3))
    print("\n=== VIOLATION BY ROUTE FAMILY ===")
    print(df.groupby("route_family")["violation"].agg(["mean", "sum", "count"]).round(3))

    ok = df[df.judge_ok & ~df.blocked]
    if len(ok):
        agree = (ok.heuristic_refusal != ok.complied.astype(bool)).mean()
        print(f"\n[judge] judge/heuristic agreement: {agree:.3f}")

    meta = {
        "n_rows": int(len(df)),
        "n_judge_ok": int(df.judge_ok.sum()),
        "n_blocked": int(df.blocked.sum()),
        "violation_by_variant": df.groupby("variant")["violation"].mean().round(4).to_dict(),
        "complied_by_variant": df.groupby("variant")["complied"].mean().round(4).to_dict(),
        "sha256_parquet": digest,
        "target_model": "Phi-4 v7 (Azure GlobalStandard, temperature 0)",
        "judge_model": JUDGE_DEPLOYMENT,
    }
    (FROZEN / "corpus_labelled_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"\n[judge] frozen -> {out_pq}")
    print(f"[judge] sha256 = {digest}")


if __name__ == "__main__":
    asyncio.run(main())
