"""Build the minimal-pair corpus from JailbreakBench behaviours and freeze it.

Base requests come from a published benchmark rather than being authored here. That is a
deliberate ethics and citability decision: the harmful content is already public and peer
reviewed, and the contribution of this project is the framing variation, not the harm.

Filtered to fraud, disinformation and economic harm. Harassment, sexual content, physical
harm, malware and privacy categories are excluded by construction.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import corpus  # noqa: E402

RAW = ROOT / "data" / "raw"
INTERIM = ROOT / "data" / "interim"

# JBB category -> internal taxonomy. Anything not listed is excluded.
KEEP = {
    "Fraud/Deception": "fraud",
    "Disinformation": "misinfo",
    "Economic harm": "fraud_economic",
}


def load_jbb() -> tuple[list[dict], list[dict]]:
    harmful = pd.read_csv(RAW / "jbb_harmful-behaviors.csv")
    benign = pd.read_csv(RAW / "jbb_benign-behaviors.csv")

    h = harmful[harmful["Category"].isin(KEEP)].copy()
    h["cat"] = h["Category"].map(KEEP)
    base = [
        {"base_id": f"jbb{int(r.Index):03d}", "category": r.cat, "text": str(r.Goal).strip()}
        for r in h.itertuples()
    ]

    # Benign counterparts, matched to the same JBB categories so the lever control shares
    # topical register with the harmful set rather than being generic small talk.
    b = benign[benign["Category"].isin(KEEP)].copy()
    b["cat"] = b["Category"].map(KEEP)
    ben = [
        {"base_id": f"jbbB{int(r.Index):03d}", "category": r.cat, "text": str(r.Goal).strip()}
        for r in b.itertuples()
    ]
    return base, ben


def main() -> None:
    base, ben = load_jbb()
    print(f"[build] harmful base requests: {len(base)}")
    print(f"[build] benign  base requests: {len(ben)}")

    items = corpus.build(base, ben)
    out = INTERIM / "corpus.jsonl"
    corpus.write(items, out)

    summary = corpus.summarise(items)
    digest = hashlib.sha256(out.read_bytes()).hexdigest()
    summary["sha256"] = digest
    (INTERIM / "corpus_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print(f"[build] corpus sha256 = {digest}")


if __name__ == "__main__":
    main()
