from __future__ import annotations

from typing import Any, Dict, List, Sequence

import numpy as np
import torch

__all__ = [
    "kl_divergence_samples",
    "cross_entropy_samples",
    "summarize",
    "alignment_stats",
]


def kl_divergence_samples(
    kl_fn: Any,
    source: torch.Tensor,
    target: torch.Tensor,
    batch_size: int,
    device: torch.device,
    max_draws: int = 300,
    seed: int = 0,
) -> List[float]:
    """Symmetric KL between random source / target embedding mini-batches."""
    if not len(source) or not len(target):
        return []
    bs = max(2, int(batch_size))
    n_draw = min(max_draws, max(1, min(len(source), len(target)) // bs))
    rng = np.random.RandomState(seed)
    values: List[float] = []
    with torch.no_grad():
        for _ in range(n_draw):
            si = torch.from_numpy(rng.randint(0, len(source), size=bs))
            ti = torch.from_numpy(rng.randint(0, len(target), size=bs))
            a = source[si].to(device)
            b = target[ti].to(device)
            values.append(float((kl_fn(a, b) + kl_fn(b, a)).detach().cpu()))
    return values


def cross_entropy_samples(
    source_probs: torch.Tensor,
    target_probs: torch.Tensor,
    max_pairs: int = 2000,
    seed: int = 1,
) -> List[float]:
    """H(source, target) over random pairs of per-image class distributions."""
    if not len(source_probs) or not len(target_probs):
        return []
    p = source_probs.clamp_min(1e-9)
    q = target_probs.clamp_min(1e-9)
    rng = np.random.RandomState(seed)
    n = min(max_pairs, max(1, min(len(p), len(q))))
    pi = torch.from_numpy(rng.randint(0, len(p), size=n))
    qi = torch.from_numpy(rng.randint(0, len(q), size=n))
    return (-(p[pi] * q[qi].log()).sum(dim=-1)).tolist()


def summarize(name: str, values: Sequence[float]) -> Dict[str, float]:
    a = np.asarray(values, dtype=np.float64)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return {
            f"{name}_mean": 0.0, f"{name}_median": 0.0, f"{name}_std": 0.0,
            f"{name}_p10": 0.0, f"{name}_p90": 0.0, f"{name}_n": 0,
        }
    return {
        f"{name}_mean": float(a.mean()),
        f"{name}_median": float(np.median(a)),
        f"{name}_std": float(a.std()),
        f"{name}_p10": float(np.percentile(a, 10)),
        f"{name}_p90": float(np.percentile(a, 90)),
        f"{name}_n": int(a.size),
    }


def _subsample(a: np.ndarray, cap: int, seed: int) -> np.ndarray:
    if len(a) <= cap:
        return a
    return a[np.random.RandomState(seed).choice(len(a), size=cap, replace=False)]


def _sq_dists(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    d = (a * a).sum(1)[:, None] + (b * b).sum(1)[None, :] - 2.0 * a @ b.T
    return np.clip(d, 0.0, None)


def _mmd_rbf(s: np.ndarray, t: np.ndarray) -> float:
    d_st = _sq_dists(s, t)
    sigma2 = float(np.median(d_st))
    if sigma2 <= 0:
        return 0.0
    k = lambda d: np.exp(-d / sigma2)
    return float(k(_sq_dists(s, s)).mean() + k(_sq_dists(t, t)).mean() - 2 * k(d_st).mean())


def _domain_auc(s: np.ndarray, t: np.ndarray) -> float:
    """AUC of a source-vs-target classifier: 0.5 = aligned, 1.0 = separable."""
    if len(s) < 4 or len(t) < 4:
        return 0.0
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import roc_auc_score
        from sklearn.model_selection import StratifiedKFold, cross_val_predict
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        x = np.concatenate([s, t], axis=0)
        y = np.concatenate([np.zeros(len(s)), np.ones(len(t))])
        folds = StratifiedKFold(
            n_splits=min(5, len(s), len(t)), shuffle=True, random_state=0
        )
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
        proba = cross_val_predict(clf, x, y, cv=folds, method="predict_proba")[:, 1]
        return float(roc_auc_score(y, proba))
    except Exception as e:
        print(f"[WARN] domain AUC failed: {e}")
        return 0.0


def alignment_stats(
    source: np.ndarray, target: np.ndarray, cap_per_domain: int = 1500
) -> Dict[str, float]:
    """Feature-level source / target alignment.

    ``align_gap``: centroid distance normalised by the pooled feature spread;
    ``align_mmd``: RBF MMD^2 (median-heuristic bandwidth);
    ``align_auc``: cross-validated domain-classifier AUC.
    """
    s = np.asarray(source, dtype=np.float64).reshape(len(source), -1)
    t = np.asarray(target, dtype=np.float64).reshape(len(target), -1)
    if not len(s) or not len(t):
        return {}
    s = _subsample(s, cap_per_domain, seed=0)
    t = _subsample(t, cap_per_domain, seed=1)
    spread = float(np.sqrt(0.5 * (s.var(axis=0).sum() + t.var(axis=0).sum())))
    gap = float(np.linalg.norm(s.mean(axis=0) - t.mean(axis=0)))
    return {
        "align_gap": gap / spread if spread > 0 else 0.0,
        "align_mmd": _mmd_rbf(s, t),
        "align_auc": _domain_auc(s, t),
    }
