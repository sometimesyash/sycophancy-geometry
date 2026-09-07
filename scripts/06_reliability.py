"""Label reliability and sampling variance. Both are irrecoverable after access ends.

1. INTER-JUDGE AGREEMENT
   The headline result is a null (sycophancy never succeeds). A null is only as credible as
   the label process behind it, so a second independent judge re-labels the corpus and we
   report Cohen's kappa. Without this, "0/444" is an assertion; with it, it is a measurement.

2. SAMPLING VARIANCE
   All generation ran at temperature 0 for reproducibility. That leaves open whether the
   observed success rates are stable or an artefact of greedy decoding. This resamples the
   discriminating framings at temperature 0.7 so the rates carry an interval.
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import azclient  # noqa: E402
import judge  # noqa: E402

INTERIM = ROOT / "data" / "interim"
FROZEN = ROOT / "data" / "frozen"
JUDGE_B = "gpt-56-sol"       # independent second judge
N_SAMPLES = 3
TEMP = 0.7


def _load_frozen() -> pd.DataFrame:
    frames = []
    for name, src in [
        ("single", "corpus_labelled.parquet"),
        ("multiturn", "corpus_mt_labelled.parquet"),
        ("expansion", "corpus_exp_labelled.parquet"),
    ]:
        p = FROZEN / src
        if p.exists():
            d = pd.read_parquet(p)
            d["corpus"] = name
            frames.append(d)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _prompt_of(row) -> str:
    # Multi-turn rows judge against the original request, not the pressure turn.
    if "turn1_user" in row and isinstance(row.get("turn1_user"), str) and row["turn1_user"]:
        return row["turn1_user"]
    return row.get("prompt", "")


async def judge_agreement(df: pd.DataFrame) -> None:
    live = df[(~df.blocked) & df.judge_ok & df.response.astype(bool)]
    print(f"[rel] second-judge pass over {len(live)} rows")

    jobs = [
        azclient.Job(
            rid=r.rid, deployment=JUDGE_B,
            messages=judge.build_messages(_prompt_of(r._asdict()), r.response),
            max_tokens=2000, temperature=None,
        )
        for r in live.itertuples()
    ]
    out = INTERIM / "judgements_judgeB.jsonl"
    await azclient.run_jobs(jobs, out, concurrency=24)

    b = {}
    for rec in azclient.read_jsonl(out):
        if rec.get("ok"):
            p = judge.parse(rec.get("content", ""))
            if p.get("parse_ok"):
                b[rec["rid"]] = p

    merged = live[live.rid.isin(b)].copy()
    merged["violation_b"] = merged.rid.map(lambda r: b[r]["violation"])
    merged["complied_b"] = merged.rid.map(lambda r: b[r]["complied"])

    def kappa(a: pd.Series, c: pd.Series) -> float:
        a, c = a.astype(bool), c.astype(bool)
        po = (a == c).mean()
        pe = a.mean() * c.mean() + (1 - a.mean()) * (1 - c.mean())
        return float((po - pe) / (1 - pe)) if pe < 1 else 1.0

    kv = kappa(merged.violation, merged.violation_b)
    kc = kappa(merged.complied, merged.complied_b)
    av = float((merged.violation.astype(bool) == merged.violation_b.astype(bool)).mean())
    ac = float((merged.complied.astype(bool) == merged.complied_b.astype(bool)).mean())

    print(f"\n[rel] n={len(merged)}")
    print(f"[rel] violation: agreement={av:.4f}  kappa={kv:.4f}")
    print(f"[rel] complied : agreement={ac:.4f}  kappa={kc:.4f}")

    disagree = merged[merged.violation.astype(bool) != merged.violation_b.astype(bool)]
    print(f"[rel] disagreements: {len(disagree)}")

    merged[["rid", "corpus", "variant", "violation", "violation_b", "complied", "complied_b"]].to_parquet(
        FROZEN / "judge_agreement.parquet", index=False
    )
    (FROZEN / "judge_agreement_meta.json").write_text(
        json.dumps(
            {"judge_a": "gpt-5-4", "judge_b": JUDGE_B, "n": int(len(merged)),
             "violation_agreement": av, "violation_kappa": kv,
             "complied_agreement": ac, "complied_kappa": kc,
             "n_disagreements": int(len(disagree))},
            indent=2,
        ),
        encoding="utf-8",
    )


async def sampling_variance(df: pd.DataFrame) -> None:
    """Resample the discriminating framings at temperature 0.7."""
    exp = df[df.corpus == "expansion"]
    if exp.empty:
        print("[var] no expansion corpus, skipping")
        return

    rate = exp.groupby("framing")["violation"].mean().sort_values(ascending=False)
    keep = list(rate.head(4).index) + ["syc_expertise", "syc_pressure", "neutral_plain"]
    keep = [k for k in dict.fromkeys(keep) if k in set(exp.framing)]
    sub = exp[exp.framing.isin(keep)]
    print(f"[var] resampling framings {keep} over {len(sub)} prompts x {N_SAMPLES}")

    jobs = []
    for r in sub.itertuples():
        for s in range(N_SAMPLES):
            jobs.append(
                azclient.Job(
                    rid=f"{r.rid}::s{s}", deployment=r.target_model,
                    messages=[{"role": "user", "content": r.prompt}],
                    max_tokens=700, temperature=TEMP,
                    meta={"base_rid": r.rid, "framing": r.framing, "target": r.target_model},
                )
            )

    gen_out = INTERIM / "generations_temp.jsonl"
    await azclient.run_jobs(jobs, gen_out, concurrency=32)
    gens = {g["rid"]: g for g in azclient.read_jsonl(gen_out)}

    pmap = {r.rid: r.prompt for r in sub.itertuples()}
    jjobs = [
        azclient.Job(rid=k, deployment="gpt-5-4",
                     messages=judge.build_messages(pmap[g["meta"]["base_rid"]], g.get("content", "")),
                     max_tokens=2000, temperature=None, meta=g["meta"])
        for k, g in gens.items() if g.get("ok")
    ]
    jout = INTERIM / "judgements_temp.jsonl"
    print(f"[var] judging {len(jjobs)}")
    await azclient.run_jobs(jjobs, jout, concurrency=24)
    judged = {j["rid"]: j for j in azclient.read_jsonl(jout)}

    rows = []
    for k, g in gens.items():
        if not g.get("ok"):
            continue
        jj = judged.get(k)
        p = judge.parse(jj.get("content", "")) if jj and jj.get("ok") else {"parse_ok": False}
        rows.append(
            {"rid": k, "base_rid": g["meta"]["base_rid"], "framing": g["meta"]["framing"],
             "target_model": g["meta"]["target"], "violation": p.get("violation"),
             "complied": p.get("complied"), "judge_ok": bool(p.get("parse_ok"))}
        )

    v = pd.DataFrame(rows)
    v.to_parquet(FROZEN / "sampling_variance.parquet", index=False)
    print("\n=== TEMP 0.7 VIOLATION RATE BY FRAMING x MODEL ===")
    print(v.pivot_table(index="framing", columns="target_model", values="violation", aggfunc="mean").round(3))


async def main() -> None:
    df = _load_frozen()
    if df.empty:
        print("[rel] no frozen corpora found")
        return
    print(f"[rel] loaded {len(df)} rows across {df.corpus.nunique()} corpora")
    await judge_agreement(df)
    await sampling_variance(df)


if __name__ == "__main__":
    asyncio.run(main())
