"""Geometry analysis: does activation structure predict capitulation? NO AZURE, NO GPU.

Runs entirely from cached activations, so this is reproducible indefinitely.

Comparator ladder, weakest to strongest. The point of the ladder is that a geometric
feature only earns its latency cost if it beats what sits below it:

  1. majority          floor
  2. tfidf             surface text only
  3. azure_embed       strong text-only features (cached before access ended)
  4. raw_pooled        linear probe on raw hidden states, single layer
  5. norms             trivial magnitude statistics
  6. lid               Local Intrinsic Dimensionality across layers
  7. curvature         token-trajectory turning angles
  8. lid+curv          the proposal's hybrid feature set
  9. all_geometric     everything geometric combined

Splitting is GROUPED BY base question or base request, never random. Random splitting would
place minimal-pair twins on both sides of the boundary, and the classifier would score by
memorising the base item rather than by detecting the phenomenon. That is the single most
likely way to produce an inflated, wrong result.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import geometry  # noqa: E402

FROZEN = ROOT / "data" / "frozen"
ACTS = ROOT / "data" / "activations"
RESULTS = ROOT / "results"

CORPUS_SRC = {
    "factual": ("corpus_fc_labelled.parquet", "capitulated", "qid"),
    "safety_exp": ("corpus_exp_labelled.parquet", "violation", "base_id"),
    "safety_mt": ("corpus_mt_labelled.parquet", "violation", "base_id"),
    "safety_single": ("corpus_labelled.parquet", "violation", "base_id"),
}


def grouped_split(groups: np.ndarray, y: np.ndarray, test_frac: float = 0.3, seed: int = 0):
    """Split by group, greedily balancing positive rate across the two sides."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    rng.shuffle(uniq)

    rate = {g: y[groups == g].mean() for g in uniq}
    order = sorted(uniq, key=lambda g: -rate[g])

    target = int(round(len(uniq) * test_frac))
    test_groups: list = []
    tr_pos = te_pos = 0.0
    for g in order:
        if len(test_groups) < target and (te_pos <= tr_pos * test_frac / (1 - test_frac)):
            test_groups.append(g)
            te_pos += rate[g]
        else:
            tr_pos += rate[g]
    test_mask = np.isin(groups, test_groups)
    return ~test_mask, test_mask


def evaluate(name, Xtr, ytr, Xte, yte, results: list):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    if len(np.unique(ytr)) < 2 or len(np.unique(yte)) < 2:
        print(f"  {name:<16} SKIPPED (single class present)")
        return

    for clf_name, clf in [
        ("logreg", make_pipeline(StandardScaler(with_mean=False),
                                 LogisticRegression(max_iter=2000, class_weight="balanced"))),
        ("rf", RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                      random_state=0, n_jobs=-1)),
    ]:
        clf.fit(Xtr, ytr)
        p = clf.predict_proba(Xte)[:, 1]
        pred = (p >= 0.5).astype(int)

        auc = roc_auc_score(yte, p)
        ap = average_precision_score(yte, p)
        f1 = f1_score(yte, pred, zero_division=0)

        # TPR at 1% FPR. In deployment the attack base rate is tiny, so false positives
        # dominate cost and headline accuracy is close to meaningless.
        neg = p[yte == 0]
        thr = np.quantile(neg, 0.99) if len(neg) else 1.0
        tpr = float((p[yte == 1] >= thr).mean()) if (yte == 1).any() else 0.0

        results.append({"features": name, "clf": clf_name, "roc_auc": auc,
                        "avg_precision": ap, "f1": f1, "tpr_at_1pct_fpr": tpr,
                        "n_train": len(ytr), "n_test": len(yte),
                        "pos_train": int(ytr.sum()), "pos_test": int(yte.sum()),
                        "n_features": Xtr.shape[1]})
        print(f"  {name:<16} {clf_name:<7} auc={auc:.3f} ap={ap:.3f} f1={f1:.3f} tpr@1%fpr={tpr:.3f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="phi4-mini")
    ap.add_argument("--corpus", default="factual", choices=list(CORPUS_SRC))
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    src, label_col, group_col = CORPUS_SRC[args.corpus]
    df = pd.read_parquet(FROZEN / src)

    npz_path = ACTS / f"{args.model}__{args.corpus}.npz"
    if not npz_path.exists():
        sys.exit(f"missing {npz_path}\nrun: python scripts/10_extract_activations.py "
                 f"--model {args.model} --corpus {args.corpus}")

    z = np.load(npz_path, allow_pickle=True)
    rids = list(z["rids"])
    pooled, per_token = z["pooled"], z["per_token"]

    df = df.set_index("rid").loc[rids].reset_index()
    y = df[label_col].fillna(False).astype(bool).astype(int).values
    groups = df[group_col].values

    print(f"[analyse] {len(df)} items | positives {y.sum()} ({y.mean():.3f}) | "
          f"{len(np.unique(groups))} groups")
    if y.sum() < 10:
        sys.exit(f"only {y.sum()} positives; not enough signal to model")

    tr, te = grouped_split(groups, y, seed=args.seed)
    print(f"[analyse] train {tr.sum()} (pos {y[tr].sum()}) | test {te.sum()} (pos {y[te].sum()})")

    results: list = []

    # 1. majority floor
    results.append({"features": "majority", "clf": "-", "roc_auc": 0.5,
                    "avg_precision": float(y[te].mean()), "f1": 0.0,
                    "tpr_at_1pct_fpr": 0.0, "n_train": int(tr.sum()),
                    "n_test": int(te.sum()), "pos_train": int(y[tr].sum()),
                    "pos_test": int(y[te].sum()), "n_features": 0})

    # 2. tfidf on the model-visible text
    from sklearn.feature_extraction.text import TfidfVectorizer
    text_col = "turn3_user" if "turn3_user" in df else "prompt"
    tf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), sublinear_tf=True)
    Xtr_t = tf.fit_transform(df[text_col].astype(str).values[tr])
    Xte_t = tf.transform(df[text_col].astype(str).values[te])
    evaluate("tfidf", Xtr_t, y[tr], Xte_t, y[te], results)

    # 3. Azure embeddings, captured before access ended
    emb_path = FROZEN / "embeddings_azure.npz"
    if emb_path.exists():
        e = np.load(emb_path, allow_pickle=True)
        emap = {r: i for i, r in enumerate(e["rids"])}
        hit = np.array([r in emap for r in rids])
        if hit.mean() > 0.5:
            E = np.zeros((len(rids), e["emb"].shape[1]), dtype=np.float32)
            for i, r in enumerate(rids):
                if r in emap:
                    E[i] = e["emb"][emap[r]]
            evaluate("azure_embed", E[tr], y[tr], E[te], y[te], results)

    # 4. raw pooled hidden states at the middle layer. This is the comparator that
    #    geometric features most need to beat.
    mid = pooled.shape[1] // 2
    R = pooled[:, mid, :].astype(np.float32)
    evaluate("raw_pooled", R[tr], y[tr], R[te], y[te], results)

    # 5. norms
    N = geometry.norm_features(pooled)
    evaluate("norms", N[tr], y[tr], N[te], y[te], results)

    # 6. LID, reference set fitted on train only
    print("[analyse] computing LID ...")
    L = geometry.lid_per_layer(pooled, np.where(tr)[0], k=args.k)
    evaluate("lid", L[tr], y[tr], L[te], y[te], results)

    # 7. curvature
    print("[analyse] computing curvature ...")
    C = geometry.curvature_per_layer(per_token)
    Cf = C.reshape(C.shape[0], -1)
    evaluate("curvature", Cf[tr], y[tr], Cf[te], y[te], results)

    # 8. the proposal's hybrid
    LC = np.concatenate([L, Cf], axis=1)
    evaluate("lid+curv", LC[tr], y[tr], LC[te], y[te], results)

    # 9. everything geometric
    A = np.concatenate([L, Cf, N], axis=1)
    evaluate("all_geometric", A[tr], y[tr], A[te], y[te], results)

    # ---- per-layer sweep: the early-exit curve --------------------------------------
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    sweep = []
    for li in range(pooled.shape[1]):
        Xl = pooled[:, li, :].astype(np.float32)
        try:
            m = make_pipeline(StandardScaler(),
                              LogisticRegression(max_iter=1500, class_weight="balanced"))
            m.fit(Xl[tr], y[tr])
            a = roc_auc_score(y[te], m.predict_proba(Xl[te])[:, 1])
        except Exception:
            a = float("nan")
        sweep.append({"layer": li, "roc_auc_raw": a, "lid_auc": float("nan")})

    for li in range(L.shape[1]):
        try:
            m = make_pipeline(StandardScaler(),
                              LogisticRegression(max_iter=1500, class_weight="balanced"))
            m.fit(L[tr, li : li + 1], y[tr])
            sweep[li]["lid_auc"] = roc_auc_score(y[te], m.predict_proba(L[te, li : li + 1])[:, 1])
        except Exception:
            pass

    RESULTS.mkdir(parents=True, exist_ok=True)
    tag = f"{args.model}__{args.corpus}"
    res_df = pd.DataFrame(results).sort_values("roc_auc", ascending=False)
    res_df.to_csv(RESULTS / f"comparison__{tag}.csv", index=False)
    pd.DataFrame(sweep).to_csv(RESULTS / f"layer_sweep__{tag}.csv", index=False)

    print("\n=== RANKED ===")
    print(res_df[["features", "clf", "roc_auc", "avg_precision", "f1", "tpr_at_1pct_fpr"]].to_string(index=False))

    best_raw = max((s["roc_auc_raw"] for s in sweep if s["roc_auc_raw"] == s["roc_auc_raw"]), default=0)
    best_layer = next((s["layer"] for s in sweep if s["roc_auc_raw"] == best_raw), -1)
    print(f"\n[analyse] best raw-probe layer {best_layer}/{pooled.shape[1]-1} auc={best_raw:.3f}")
    if best_layer >= 0:
        print(f"[analyse] early-exit depth = {best_layer/(pooled.shape[1]-1):.1%} of the network")

    (RESULTS / f"summary__{tag}.json").write_text(json.dumps({
        "model": args.model, "corpus": args.corpus,
        "n_items": len(df), "n_positive": int(y.sum()),
        "best_features": res_df.iloc[0]["features"],
        "best_auc": float(res_df.iloc[0]["roc_auc"]),
        "best_raw_layer": int(best_layer), "best_raw_auc": float(best_raw),
    }, indent=2))


if __name__ == "__main__":
    main()
