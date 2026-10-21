from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np
import torch

__all__ = [
    "alignment_stats",
    "cross_entropy_samples",
    "kl_divergence_samples",
    "summarize",
]


def summarize(name: str, values: Sequence[float]) -> Dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {
            f"{name}_mean": 0.0,
            f"{name}_median": 0.0,
            f"{name}_std": 0.0,
            f"{name}_p10": 0.0,
            f"{name}_p90": 0.0,
            f"{name}_n": 0,
        }
    return {
        f"{name}_mean": float(arr.mean()),
        f"{name}_median": float(np.median(arr)),
        f"{name}_std": float(arr.std()),
        f"{name}_p10": float(np.percentile(arr, 10)),
        f"{name}_p90": float(np.percentile(arr, 90)),
        f"{name}_n": int(arr.size),
    }


def kl_divergence_samples(
    kl_module: torch.nn.Module,
    source: torch.Tensor,
    target: torch.Tensor,
    batch_size: int,
    device: torch.device,
    max_draws: int = 300,
    seed: int = 0,
) -> List[float]:
    if len(source) == 0 or len(target) == 0:
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
            values.append(float((kl_module(a, b) + kl_module(b, a)).detach().cpu()))
    return values


def cross_entropy_samples(
    source_probs: torch.Tensor,
    target_probs: torch.Tensor,
    max_pairs: int = 2000,
    seed: int = 1,
) -> List[float]:
    if len(source_probs) == 0 or len(target_probs) == 0:
        return []
    src = source_probs.clamp_min(1e-9)
    tgt = target_probs.clamp_min(1e-9)
    rng = np.random.RandomState(seed)
    n = min(max_pairs, max(1, min(len(src), len(tgt))))
    si = torch.from_numpy(rng.randint(0, len(src), size=n))
    ti = torch.from_numpy(rng.randint(0, len(tgt), size=n))
    ce = -(src[si] * tgt[ti].log()).sum(dim=-1)
    return ce.tolist()


def _subsample(x: np.ndarray, cap: int, seed: int) -> np.ndarray:
    if len(x) <= cap:
        return x
    idx = np.random.RandomState(seed).choice(len(x), size=cap, replace=False)
    return x[idx]


def _rbf_mmd2(x: np.ndarray, y: np.ndarray) -> float:
    pooled = np.concatenate([x, y], axis=0)
    dists = np.sum((pooled[:, None, :] - pooled[None, :, :]) ** 2, axis=-1)
    median = float(np.median(dists[dists > 0])) if np.any(dists > 0) else 1.0
    gamma = 1.0 / max(median, 1e-09)
    n, m = (len(x), len(y))
    kxx = np.exp(-gamma * dists[:n, :n])
    kyy = np.exp(-gamma * dists[n:, n:])
    kxy = np.exp(-gamma * dists[:n, n:])
    np.fill_diagonal(kxx, 0.0)
    np.fill_diagonal(kyy, 0.0)
    term_x = kxx.sum() / max(n * (n - 1), 1)
    term_y = kyy.sum() / max(m * (m - 1), 1)
    term_xy = kxy.mean() if kxy.size else 0.0
    return float(term_x + term_y - 2.0 * term_xy)


def _separability_auc(x: np.ndarray, y: np.ndarray) -> float:
    direction = y.mean(axis=0) - x.mean(axis=0)
    norm = float(np.linalg.norm(direction))
    if norm < 1e-12:
        return 0.5
    direction = direction / norm
    scores = np.concatenate([x @ direction, y @ direction])
    labels = np.concatenate([np.zeros(len(x)), np.ones(len(y))])
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(labels, scores))


def alignment_stats(
    source: np.ndarray,
    target: np.ndarray,
    cap: int = 1000,
    seed: int = 0,
) -> Dict[str, float]:
    src = np.asarray(source, dtype=np.float64).reshape(len(source), -1)
    tgt = np.asarray(target, dtype=np.float64).reshape(len(target), -1)
    if src.size == 0 or tgt.size == 0:
        return {
            "align_gap": 0.0,
            "align_mmd": 0.0,
            "align_auc": 0.5,
            "align_n_source": float(len(src)),
            "align_n_target": float(len(tgt)),
        }
    gap = float(np.linalg.norm(src.mean(axis=0) - tgt.mean(axis=0)))
    sub_src = _subsample(src, cap, seed)
    sub_tgt = _subsample(tgt, cap, seed + 1)
    return {
        "align_gap": gap,
        "align_mmd": _rbf_mmd2(sub_src, sub_tgt),
        "align_auc": _separability_auc(src, tgt),
        "align_n_source": float(len(src)),
        "align_n_target": float(len(tgt)),
    }
