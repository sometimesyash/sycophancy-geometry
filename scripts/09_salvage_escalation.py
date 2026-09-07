"""Rebuild the frozen escalation corpus from interim checkpoints.

The escalation script only writes its frozen parquet after every arm finishes. The
geometry arm (the one that matters for RQ2) completes well before the scaling arm, so this
salvages whatever is on disk into the frozen artefacts at any point.

Safe to run repeatedly. Reads only checkpoint files; never calls the network.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import capitulation  # noqa: E402

INTERIM = ROOT / "data" / "interim"
FROZEN = ROOT / "data" / "frozen"

ARMS = {"geo": "geometry", "scale": "scaling"}
LEVER_FAMILY = {
    "fc_authority": "false_premise",
    "fc_expertise_override": "false_premise",
    "fc_flat_contradiction": "false_premise",
    "fc_disappointment": "social_pressure",
    "fc_neutral_reask": "neutral",
}


def load_arm(tag: str) -> pd.DataFrame:
    rows = []
    for rnd in range(3):
        gen_p = INTERIM / f"fc2_gen_{tag}_r{rnd}.jsonl"
        jud_p = INTERIM / f"fc2_judge_{tag}_r{rnd}.jsonl"
        if not (gen_p.exists() and jud_p.exists()):
            continue

        gens = {}
        for line in gen_p.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                if r.get("ok"):
                    gens[r["rid"]] = r.get("content", "")

        for line in jud_p.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if not r.get("ok"):
                continue
            p = capitulation.parse(r.get("content", ""))
            if not p.get("parse_ok"):
                continue
            cid = r["rid"]
            parts = cid.split("::")
            if len(parts) < 4:
                continue
            qid, lever, model, samp = parts[0], parts[1], parts[2], parts[3]
            rows.append({
                "conv_id": cid, "qid": qid, "lever_name": lever,
                "route_family": LEVER_FAMILY.get(lever, "unknown"),
                "variant": "neutral" if lever == "fc_neutral_reask" else "syc",
                "target_model": model, "sample": int(samp.lstrip("s") or 0),
                "round": rnd, "capitulated": bool(p["capitulated"]),
                "held_firm": bool(p.get("held_firm")), "hedged": bool(p.get("hedged")),
                "response": gens.get(cid, ""), "arm": ARMS[tag],
            })
    return pd.DataFrame(rows)


def main() -> None:
    frames = [d for d in (load_arm(t) for t in ARMS) if len(d)]
    if not frames:
        sys.exit("no escalation checkpoints found")
    per_round = pd.concat(frames, ignore_index=True)
    print(f"[salvage] {len(per_round)} judged round-records")

    # One row per conversation: capitulated if it broke in ANY round, and at which round.
    def collapse(g: pd.DataFrame) -> pd.Series:
        cap = g[g.capitulated]
        return pd.Series({
            "capitulated": bool(len(cap)),
            "capitulated_at": int(cap["round"].min()) if len(cap) else None,
            "n_rounds": int(g["round"].max()) + 1,
            "final_response": g.sort_values("round").iloc[-1]["response"],
        })

    keys = ["conv_id", "qid", "lever_name", "route_family", "variant",
            "target_model", "sample", "arm"]
    conv = per_round.groupby(keys, dropna=False).apply(collapse, include_groups=False).reset_index()

    FROZEN.mkdir(parents=True, exist_ok=True)
    out_pq = FROZEN / "corpus_fc_escalation.parquet"
    conv.to_parquet(out_pq, index=False)
    conv.to_csv(FROZEN / "corpus_fc_escalation.csv", index=False)
    per_round.to_parquet(FROZEN / "corpus_fc_escalation_rounds.parquet", index=False)

    # Per-item propensity over the sampled repeats: a continuous target for geometry.
    prop = (conv[conv.variant == "syc"]
            .groupby(["arm", "target_model", "qid", "lever_name"])["capitulated"]
            .mean().reset_index().rename(columns={"capitulated": "propensity"}))
    prop.to_parquet(FROZEN / "capitulation_propensity.parquet", index=False)

    print("\n=== CAPITULATION BY MODEL (sycophantic levers, any round) ===")
    print(conv[conv.variant == "syc"].groupby("target_model")["capitulated"]
          .agg(["mean", "sum", "count"]).round(3))
    print("\n=== NEUTRAL CONTROL ===")
    print(conv[conv.variant == "neutral"].groupby("target_model")["capitulated"]
          .agg(["mean", "sum", "count"]).round(3))
    print("\n=== BY ROUND (syc) ===")
    print(per_round[per_round.variant == "syc"].groupby(["target_model", "round"])["capitulated"]
          .agg(["mean", "sum", "count"]).round(3))
    print("\n=== BY LEVER (syc) ===")
    print(conv[conv.variant == "syc"].groupby("lever_name")["capitulated"]
          .agg(["mean", "sum", "count"]).round(3))
    print("\n=== POSITIVES AVAILABLE FOR GEOMETRY ===")
    geo = conv[(conv.arm == "geometry") & (conv.variant == "syc")]
    for m, g in geo.groupby("target_model"):
        print(f"  {m:<18} n={len(g):<5} positives={int(g.capitulated.sum())}")

    meta = {
        "n_conversations": int(len(conv)),
        "n_round_records": int(len(per_round)),
        "arms_present": sorted(conv["arm"].unique().tolist()),
        "models": sorted(conv["target_model"].unique().tolist()),
        "positives_geometry_arm": {
            m: int(g.capitulated.sum()) for m, g in geo.groupby("target_model")
        },
        "sha256_parquet": hashlib.sha256(out_pq.read_bytes()).hexdigest(),
        "note": "Rebuilt from interim checkpoints; the scaling arm may be partial.",
    }
    (FROZEN / "corpus_fc_escalation_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"\n[salvage] frozen -> {out_pq}")
    print(f"[salvage] sha256 = {meta['sha256_parquet']}")


if __name__ == "__main__":
    main()
