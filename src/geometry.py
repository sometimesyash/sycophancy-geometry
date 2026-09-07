"""Geometric features from cached activations. NO AZURE, NO GPU.

Implements the two features named in the proposal, plus the baselines that decide whether
they earn their cost.

Local Intrinsic Dimensionality (LID)
------------------------------------
Maximum-likelihood estimator of Levina & Bickel (2004), in the form used by Amsaleg et al.
and adopted by CurvaLID:

    LID(x) = -[ (1/k) * sum_i log(r_i / r_k) ]^-1

where r_i is the distance to the i-th nearest neighbour. Intuitively it measures how many
degrees of freedom the data occupies locally: a point sitting on a low-dimensional manifold
has low LID, a point in an anomalous or poorly-modelled region has high LID.

IMPORTANT: LID is computed against a REFERENCE set drawn from training data only. Fitting
neighbours on the full corpus would leak test information through the feature itself,
which is exactly the data-snooping failure Arp et al. (USENIX 2022) document. The reference
set is fixed at fit time and reused unchanged at transform time.

Curvature
---------
Discrete curvature of the token trajectory through hidden space. For consecutive states
h_1..h_T we take unit difference vectors and measure the turning angle between them:

    theta_t = arccos( <d_t, d_{t+1}> / (|d_t| |d_{t+1}|) )

Summary statistics over theta capture how sharply the representation changes direction as
the prompt is read. A trajectory that proceeds smoothly differs from one that is repeatedly
redirected.
"""
from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# LID
# ---------------------------------------------------------------------------

def _pairwise_sq(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Squared euclidean distances, chunk-safe for modest matrices."""
    a2 = (a * a).sum(1)[:, None]
    b2 = (b * b).sum(1)[None, :]
    d = a2 + b2 - 2.0 * (a @ b.T)
    return np.maximum(d, 0.0)


def lid_mle(X: np.ndarray, reference: np.ndarray, k: int = 20,
            exclude_self: bool = False, block: int = 512) -> np.ndarray:
    """MLE LID of each row of X against `reference`.

    `exclude_self` must be True when X IS the reference set, otherwise the zero
    self-distance enters the estimate and drives it to nonsense.
    """
    X = np.ascontiguousarray(X, dtype=np.float32)
    R = np.ascontiguousarray(reference, dtype=np.float32)
    n = X.shape[0]
    out = np.zeros(n, dtype=np.float32)
    need = k + (1 if exclude_self else 0)

    for s in range(0, n, block):
        d2 = _pairwise_sq(X[s : s + block], R)
        idx = np.argpartition(d2, need, axis=1)[:, : need + 1]
        part = np.take_along_axis(d2, idx, axis=1)
        part.sort(axis=1)
        if exclude_self:
            part = part[:, 1:]
        r = np.sqrt(part[:, :k])

        # Degenerate neighbourhoods (duplicate prompts) would divide by zero.
        rk = r[:, -1:].copy()
        rk[rk <= 0] = np.finfo(np.float32).eps
        ratio = np.clip(r / rk, np.finfo(np.float32).eps, 1.0)
        s_log = np.log(ratio).mean(axis=1)
        s_log[s_log >= -1e-12] = -1e-12
        out[s : s + block] = (-1.0 / s_log).astype(np.float32)

    return out


class LIDExtractor:
    """LID with a reference set frozen at fit time, so no test information leaks in."""

    def __init__(self, k: int = 20, max_reference: int = 4000, seed: int = 0) -> None:
        self.k = k
        self.max_reference = max_reference
        self.seed = seed
        self.reference_: np.ndarray | None = None

    def fit(self, X_train: np.ndarray) -> "LIDExtractor":
        rng = np.random.default_rng(self.seed)
        X = np.asarray(X_train, dtype=np.float32)
        if X.shape[0] > self.max_reference:
            X = X[rng.choice(X.shape[0], self.max_reference, replace=False)]
        self.reference_ = X
        return self

    def transform(self, X: np.ndarray, is_reference: bool = False) -> np.ndarray:
        if self.reference_ is None:
            raise RuntimeError("call fit() first")
        return lid_mle(X, self.reference_, k=self.k, exclude_self=is_reference)


def lid_per_layer(pooled: np.ndarray, train_idx: np.ndarray, k: int = 20) -> np.ndarray:
    """LID at every layer. pooled is (n_items, n_layers, hidden) -> (n_items, n_layers)."""
    n, L, _ = pooled.shape
    out = np.zeros((n, L), dtype=np.float32)
    mask = np.zeros(n, dtype=bool)
    mask[train_idx] = True
    for li in range(L):
        Xl = pooled[:, li, :].astype(np.float32)
        ex = LIDExtractor(k=k).fit(Xl[mask])
        out[mask, li] = ex.transform(Xl[mask], is_reference=True)
        out[~mask, li] = ex.transform(Xl[~mask], is_reference=False)
    return out


# ---------------------------------------------------------------------------
# Curvature
# ---------------------------------------------------------------------------

def trajectory_curvature(traj: np.ndarray, eps: float = 1e-6) -> dict[str, float]:
    """Turning-angle statistics for one token trajectory, shape (T, hidden)."""
    H = np.asarray(traj, dtype=np.float32)
    H = H[~np.all(H == 0, axis=1)]  # drop padding rows
    if H.shape[0] < 3:
        return {"curv_mean": 0.0, "curv_std": 0.0, "curv_max": 0.0,
                "curv_sum": 0.0, "curv_final": 0.0, "path_ratio": 0.0}

    d = np.diff(H, axis=0)
    nrm = np.linalg.norm(d, axis=1, keepdims=True)
    nrm[nrm < eps] = eps
    u = d / nrm

    cos = np.clip((u[:-1] * u[1:]).sum(axis=1), -1.0, 1.0)
    theta = np.arccos(cos)

    seg = np.linalg.norm(d, axis=1).sum()
    direct = float(np.linalg.norm(H[-1] - H[0]))

    return {
        "curv_mean": float(theta.mean()),
        "curv_std": float(theta.std()),
        "curv_max": float(theta.max()),
        "curv_sum": float(theta.sum()),
        "curv_final": float(theta[-1]),
        # >1 means the trajectory wanders relative to its net displacement.
        "path_ratio": float(seg / max(direct, eps)),
    }


def curvature_per_layer(per_token: np.ndarray) -> np.ndarray:
    """Curvature stats for every item and layer.

    per_token is (n_items, n_kept_layers, T, hidden) -> (n_items, n_kept_layers, 6).
    """
    n, L, _, _ = per_token.shape
    keys = ["curv_mean", "curv_std", "curv_max", "curv_sum", "curv_final", "path_ratio"]
    out = np.zeros((n, L, len(keys)), dtype=np.float32)
    for i in range(n):
        for li in range(L):
            st = trajectory_curvature(per_token[i, li])
            out[i, li] = [st[k] for k in keys]
        if i % 200 == 0:
            print(f"[curv] {i}/{n}", flush=True)
    return out


CURV_KEYS = ["curv_mean", "curv_std", "curv_max", "curv_sum", "curv_final", "path_ratio"]


# ---------------------------------------------------------------------------
# Cheap norm-based features, included so the ablation can show whether LID and curvature
# beat trivial magnitude statistics. If they do not, that is the honest finding.
# ---------------------------------------------------------------------------

def norm_features(pooled: np.ndarray) -> np.ndarray:
    """(n_items, n_layers) L2 norms, plus layer-to-layer deltas."""
    nrm = np.linalg.norm(pooled.astype(np.float32), axis=2)
    delta = np.diff(nrm, axis=1, prepend=nrm[:, :1])
    return np.concatenate([nrm, delta], axis=1)
