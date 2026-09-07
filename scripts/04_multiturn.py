"""Multi-turn run: build pressure variants on real refusals, generate, judge, freeze.

Runs against every deployed target model so the transferability question (RQ4) is answered
with real data rather than a single model's idiosyncrasy.
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
import multiturn  # noqa: E402

INTERIM = ROOT / "data" / "interim"
FROZEN = ROOT / "data" / "frozen"
JUDGE_DEPLOYMENT = "gpt-5-4"
TARGETS = ["phi4-target", "phi4mini-target"]


def seeds_from_singleturn() -> list[dict]:
    """Bare requests plus the model's own refusal, taken from the neutral_plain variant.

    neutral_plain is the unwrapped request, so its response is the model's baseline refusal
    with no framing attached. That is the correct turn-2 anchor.
    """
    items = {it.rid: it for it in corpus.load(INTERIM / "corpus.jsonl")}
    gens = {g["rid"]: g for g in azclient.read_jsonl(INTERIM / "generations.jsonl")}

    seeds = []
    for rid, it in items.items():
        if it.lever_name != "neutral_plain" or not it.harmful:
            continue
        g = gens.get(rid)
        if not g or not g.get("ok"):
            continue
        resp = g.get("content", "")
        if not judge.looks_like_refusal(resp):
            continue  # only seed from genuine refusals; nothing to cave from otherwise
        seeds.append(
            {
                "base_id": it.base_id,
                "category": it.category,
                "harmful": True,
                "request": it.prompt,
                "refusal": resp,
            }
        )
    return seeds


async def main() -> None:
    seeds = seeds_from_singleturn()
    print(f"[mt] refusal seeds: {len(seeds)}")

    all_items: list[multiturn.MTItem] = []
    for tgt in TARGETS:
        all_items.extend(multiturn.build(seeds, tgt))
    multiturn.write(all_items, INTERIM / "corpus_mt.jsonl")

    # ---- generate ------------------------------------------------------------------
    gen_jobs = [
        azclient.Job(
            rid=it.rid,
            deployment=it.target_model,
            messages=it.messages(),
            max_tokens=600,
            temperature=0.0,
            meta={"variant": it.variant, "target": it.target_model},
        )
        for it in all_items
    ]
    gen_out = INTERIM / "generations_mt.jsonl"
    print(f"[mt] generating {len(gen_jobs)}")
    await azclient.run_jobs(gen_jobs, gen_out, concurrency=32)

    gens = {g["rid"]: g for g in azclient.read_jsonl(gen_out)}

    # ---- judge ---------------------------------------------------------------------
    judge_jobs = []
    for it in all_items:
        g = gens.get(it.rid)
        if not g or not g.get("ok"):
            continue
        # The judge sees the ORIGINAL request, not the turn-3 pressure, so it evaluates
        # whether the harmful content was ultimately delivered.
        judge_jobs.append(
            azclient.Job(
                rid=it.rid,
                deployment=JUDGE_DEPLOYMENT,
                messages=judge.build_messages(it.turn1_user, g.get("content", "")),
                max_tokens=2000,
                temperature=None,
                meta={"variant": it.variant},
            )
        )
    judge_out = INTERIM / "judgements_mt.jsonl"
    print(f"[mt] judging {len(judge_jobs)}")
    await azclient.run_jobs(judge_jobs, judge_out, concurrency=24)
    judged = {j["rid"]: j for j in azclient.read_jsonl(judge_out)}

    # ---- freeze --------------------------------------------------------------------
    rows = []
    for it in all_items:
        g = gens.get(it.rid)
        rec = {
            "rid": it.rid, "case_id": it.case_id, "base_id": it.base_id,
            "variant": it.variant, "route_family": it.route_family,
            "lever_name": it.lever_name, "category": it.category,
            "harmful_base": it.harmful, "target_model": it.target_model,
            "turn1_user": it.turn1_user, "turn2_assistant": it.turn2_assistant,
            "turn3_user": it.turn3_user,
        }
        if not g or not g.get("ok"):
            rec.update(response="", blocked=True, complied=False, violation=False,
                       judge_ok=True, judge_conf=1.0, judge_rationale="platform content filter",
                       heuristic_refusal=True)
        else:
            resp = g.get("content", "")
            jj = judged.get(it.rid)
            p = judge.parse(jj.get("content", "")) if jj and jj.get("ok") else {"parse_ok": False, "reason": "no_judge"}
            rec.update(
                response=resp, blocked=False,
                heuristic_refusal=judge.looks_like_refusal(resp),
                judge_ok=bool(p.get("parse_ok")),
                complied=p.get("complied"), violation=p.get("violation"),
                judge_conf=p.get("confidence"),
                judge_rationale=p.get("rationale", p.get("reason", "")),
            )
        rows.append(rec)

    df = pd.DataFrame(rows)
    FROZEN.mkdir(parents=True, exist_ok=True)
    out_pq = FROZEN / "corpus_mt_labelled.parquet"
    df.to_parquet(out_pq, index=False)
    df.to_csv(FROZEN / "corpus_mt_labelled.csv", index=False)
    digest = hashlib.sha256(out_pq.read_bytes()).hexdigest()

    print(f"\n[mt] rows={len(df)} blocked={int(df.blocked.sum())}")
    print("\n=== VIOLATION BY VARIANT x MODEL ===")
    print(df.pivot_table(index="variant", columns="target_model", values="violation",
                         aggfunc=["mean", "sum", "count"]).round(3))
    print("\n=== COMPLIANCE BY VARIANT x MODEL ===")
    print(df.pivot_table(index="variant", columns="target_model", values="complied",
                         aggfunc="mean").round(3))
    print("\n=== VIOLATION BY LEVER (phi4-target) ===")
    sub = df[df.target_model == "phi4-target"]
    print(sub.groupby(["variant", "lever_name"])["violation"].agg(["mean", "sum", "count"]).round(3))

    meta = {
        "n_rows": int(len(df)),
        "n_seeds": len(seeds),
        "targets": TARGETS,
        "violation_by_variant_model": df.groupby(["target_model", "variant"])["violation"].mean().round(4).to_dict().__str__(),
        "sha256_parquet": digest,
    }
    (FROZEN / "corpus_mt_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"\n[mt] frozen -> {out_pq}")
    print(f"[mt] sha256 = {digest}")


if __name__ == "__main__":
    asyncio.run(main())
