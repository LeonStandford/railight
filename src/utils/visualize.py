from __future__ import annotations
import math
import os
import textwrap
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
import numpy as np

matplotlib.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "legend.framealpha": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linestyle": "--",
        "figure.facecolor": "white",
        "axes.facecolor": "#F9F9F9",
    }
)
Point = Tuple[float, float]
History = Dict[str, List[Point]]
Config = Optional[Dict[str, Any]]
_DPI = 200
_BLUE = "tab:blue"
_RED = "tab:red"
_TAB_COLORS = list(plt.get_cmap("tab10").colors)
_PRETTY_LOSS_NAME: Dict[str, str] = {
    "total": "Total training loss",
    "pal1_loc": "First shot localisation loss",
    "pal1_conf": "First shot classification loss",
    "pal2_loc": "Second shot localisation loss",
    "pal2_conf": "Second shot classification loss",
    "enhance": "Retinex decomposition loss",
    "enhance_l1ssim": "Reflectance reconstruction loss",
    "mutual": "Mutual learning loss",
    "kl_st": "Feature alignment loss",
    "target_unsup": "Target reconstruction loss",
    "target_sup": "Target supervised detection loss",
    "pseudo": "Pseudo label loss",
    "wreg": "Weight regularisation loss",
    "entropy": "Target prediction entropy loss",
    "train_loss_epoch": "Training loss",
    "val_loss": "Validation loss",
    "target_val_loss": "Target validation loss",
}
_LOSS_SUBTITLE: Dict[str, str] = {
    "total": "Sum of every loss term below, the value actually back-propagated",
    "pal1_loc": "Box regression on the first detection head, source images only",
    "pal1_conf": "Class prediction on the first detection head, source images only",
    "pal2_loc": "Box regression on the second, refined detection head, source images only",
    "pal2_conf": "Class prediction on the second, refined detection head, source images only",
    "enhance": "Retinex split of the image into reflectance and illumination",
    "enhance_l1ssim": "Reflectance predicted inside the network versus the frozen RetinexNet",
    "mutual": "Agreement between the daylight branch and the synthetic dark branch",
    "kl_st": "Divergence between source and target features, pulls the two domains together",
    "target_unsup": "Rebuilds the target image from its reflectance and illumination, needs no labels",
    "target_sup": "Detection loss on real target labels, both heads, same recipe as the source loss",
    "pseudo": "Detection loss on boxes invented by the mean teacher for unlabelled target images",
    "wreg": "Keeps the backbone close to its pretrained weights",
    "entropy": "Pushes target predictions to be confident rather than undecided",
}
_LOSS_ORDER: Tuple[str, ...] = (
    "total",
    "pal1_loc",
    "pal1_conf",
    "pal2_loc",
    "pal2_conf",
    "enhance",
    "enhance_l1ssim",
    "mutual",
)
_EPOCH_KEYS = frozenset({"train_loss_epoch", "val_loss"})

def _wrap_subtitle(text: str, width: int = 58) -> str:
    """Wrap a one-sentence panel subtitle so it fits above the axes."""
    return "\n".join(textwrap.wrap(text, width=width))


def _run_tag(method: str, config: Config) -> str:
    parts: List[str] = []
    if method:
        parts.append(str(method))
    if config:
        bb = config.get("backbone")
        if bb:
            parts.append(f"({bb})")
        exp = config.get("exp") or config.get("num_exp")
        if exp:
            parts.append(f"Exp {exp}")
    return " | ".join(parts)

def make_charts_dir(
    charts_root: str, mode: str, architecture: str, backbone: str, num_exp: str
) -> str:
    out = os.path.join(
        str(charts_root), str(mode), str(architecture), str(backbone), str(num_exp)
    )
    os.makedirs(out, exist_ok=True)
    return out

def _config_suffix(config: Config) -> str:
    if not config:
        return ""
    return " | ".join((f"{k}={v}" for (k, v) in config.items()))

def _compose_title(method: str, subject: str, config: Config) -> str:
    title = f"{method} — {subject}"
    sub = _config_suffix(config)
    if sub:
        title += f"\n({sub})"
    return title

def _save(fig: Figure, path: str) -> str:
    fig.tight_layout()
    fig.savefig(path, dpi=_DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path

def plot_losses(
    history: History,
    out_dir: str,
    method: str = "RAILIGHT",
    config: Config = None,
    x_key: str = "iter",
) -> Optional[str]:
    if not history:
        return None

    loss_fn_keys = list(_LOSS_ORDER) + [
        "target_unsup", "target_sup", "pseudo", "kl_st", "wreg", "entropy"
    ]
    iter_panels = [k for k in loss_fn_keys if history.get(k)]

    combined = [
        ("Detection loss per epoch", "train_det_epoch", "val_loss"),
        ("Precision", "train_precision", "val_precision"),
        ("Recall", "train_recall", "val_recall"),
        ("F1 score", "train_f1", "val_f1"),
        ("mAP@0.5", "train_map", "val_map"),
        ("Target detection loss per epoch", None, "target_val_loss"),
        ("Target precision", None, "target_precision"),
        ("Target recall", None, "target_recall"),
        ("Target F1 score", None, "target_f1"),
        ("Target mAP@0.5", None, "target_map"),
    ]
    combined = [
        (lbl, tk, vk)
        for (lbl, tk, vk) in combined
        if history.get(tk) or history.get(vk)
    ]

    n = len(iter_panels) + len(combined)
    if n == 0:
        return None
    ncols = min(4, n)
    nrows = math.ceil(n / ncols)
    run_tag = _run_tag(method, config)
    run_tag_lines = run_tag.split(" | ") if run_tag else []
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(6 * ncols, 5 * nrows), squeeze=False
    )
    axes_flat = axes.flatten()

    idx = 0
    for key in iter_panels:
        ax = axes_flat[idx]
        xs, ys = zip(*history[key])
        color = _TAB_COLORS[idx % len(_TAB_COLORS)]
        if len(ys) < 12:
            ax.plot(
                xs, ys, color=color, linewidth=2.0, marker="o",
                markersize=3,
            )
        else:
            ax.plot(xs, ys, color=color, linewidth=0.8, alpha=0.30)
            ema, a, prev = [], 0.1, ys[0]
            for y in ys:
                prev = a * y + (1 - a) * prev
                ema.append(prev)
            ax.plot(xs, ema, color=color, linewidth=2.2, label="EMA(0.1)")
            ax.legend(loc="best", framealpha=0.85, fontsize=8)
        title = _PRETTY_LOSS_NAME.get(key, key.replace("_", " ").capitalize())
        subtitle = _wrap_subtitle(_LOSS_SUBTITLE.get(key, ""))
        n_subtitle_lines = len(subtitle.split("\n")) if subtitle else 0
        ax.set_title(
            title,
            fontweight="bold",
            fontsize=12,
            pad=10 + 12 * n_subtitle_lines,
        )
        if subtitle:
            ax.text(
                0.5, 1.012, subtitle,
                transform=ax.transAxes, ha="center", va="bottom",
                fontsize=8.5, style="italic", color="#444444", linespacing=1.35,
            )
        ax.set_xlabel("Iteration", fontweight="bold")
        ax.set_ylabel("Loss value", fontweight="bold")
        idx += 1

    for label, tr_key, va_key in combined:
        ax = axes_flat[idx]
        tr, va = history.get(tr_key, []), history.get(va_key, [])
        if tr:
            xs, ys = zip(*tr)
            ax.plot(
                xs, ys, color=_BLUE, linewidth=2.0, marker="o",
                markersize=4, label="train",
            )
        if va:
            xs, ys = zip(*va)
            ax.plot(
                xs, ys, color=_RED, linewidth=2.0, marker="s",
                markersize=4, label="val",
            )
        ax.set_title(
            f"{label}  (train versus validation)",
            fontweight="bold", fontsize=12, pad=10,
        )
        ax.set_xlabel("Epoch", fontweight="bold")
        ax.set_ylabel(label, fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend(loc="best", framealpha=0.9, fontsize=8)
        idx += 1

    for j in range(n, len(axes_flat)):
        axes_flat[j].set_visible(False)
    if run_tag:
        fig.suptitle(
            f"Detailed loss components — {run_tag}",
            fontsize=14, fontweight="bold", y=1.005,
        )
    fig.tight_layout(h_pad=2.5, w_pad=1.5, rect=(0, 0, 1, 0.985))
    fig.savefig(
        os.path.join(out_dir, "losses.png"),
        dpi=_DPI,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)
    return os.path.join(out_dir, "losses.png")

def plot_train_vs_val(
    train_pts: Sequence[Point],
    val_pts: Sequence[Point],
    out_dir: str,
    method: str = "RAILIGHT",
    config: Config = None,
    fname: str = "train_vs_val.png",
) -> Optional[str]:
    if not train_pts and (not val_pts):
        return None
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_facecolor("#ECECEC")
    if train_pts:
        xs, ys = zip(*train_pts)
        ax.plot(
            xs,
            ys,
            color=_BLUE,
            linewidth=2.0,
            marker="o",
            markersize=5,
            label="train (pal2 det / epoch)",
        )
    if val_pts:
        xs, ys = zip(*val_pts)
        ax.plot(
            xs,
            ys,
            color=_RED,
            linewidth=2.0,
            marker="s",
            markersize=5,
            label="val (target proxy_loss)",
        )
    ax.set_xlabel("Epoch", fontweight="bold")
    ax.set_ylabel("Loss", fontweight="bold")
    run_tag = _run_tag(method, config)
    title = "Train vs Val"
    if run_tag:
        title = f"{title}\n{run_tag}"
    ax.set_title(title, fontweight="bold", fontsize=13, pad=10)
    ax.grid(True, linestyle="--", alpha=0.5, color="white", linewidth=1.2)
    ax.set_axisbelow(True)
    ax.legend(loc="best", framealpha=0.9)
    return _save(fig, os.path.join(out_dir, fname))

def plot_train_val_metrics(
    history: History,
    out_dir: str,
    method: str = "RAILIGHT",
    config: Config = None,
    fname: str = "train_val_metrics.png",
) -> Optional[str]:
    """One figure: each metric overlays TRAIN (blue) vs VAL (red).

    Panels: Loss, Precision, Recall, F1, mAP. Train and val share the
    same axes per panel so the gap (overfitting) is visible at a glance.
    """
    panels = [
        ("Loss", "train_det_epoch", "val_loss"),
        ("Precision", "train_precision", "val_precision"),
        ("Recall", "train_recall", "val_recall"),
        ("F1 score", "train_f1", "val_f1"),
        ("mAP@0.5", "train_map", "val_map"),
    ]
    panels = [
        p for p in panels if history.get(p[1]) or history.get(p[2])
    ]
    if not panels:
        return None
    n = len(panels)
    ncols = 2
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(7.5 * ncols, 4.2 * nrows), squeeze=False
    )
    for idx, (label, tr_key, va_key) in enumerate(panels):
        ax = axes[idx // ncols][idx % ncols]
        ax.set_facecolor("#ECECEC")
        tr = history.get(tr_key, [])
        va = history.get(va_key, [])
        if tr:
            xs, ys = zip(*tr)
            ax.plot(
                xs, ys, color=_BLUE, linewidth=2.0, marker="o",
                markersize=4, label="train",
            )
        if va:
            xs, ys = zip(*va)
            ax.plot(
                xs, ys, color=_RED, linewidth=2.0, marker="s",
                markersize=4, label="val",
            )
        ax.set_title(label, fontweight="bold", fontsize=12)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(label)
        ax.grid(True, linestyle="--", alpha=0.5, color="white",
                linewidth=1.0)
        ax.set_axisbelow(True)
        ax.legend(loc="best", framealpha=0.9)
    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].set_visible(False)
    run_tag = _run_tag(method, config)
    sup = "Train vs Val — metrics"
    if run_tag:
        sup = f"{sup}  ·  {run_tag}"
    fig.suptitle(sup, fontweight="bold", fontsize=13)
    fig.subplots_adjust(top=0.90)
    return _save(fig, os.path.join(out_dir, fname))

def plot_confusion_matrix(
    cm: np.ndarray,
    out_dir: str,
    method: str = "RAILIGHT",
    config: Config = None,
    classes: Sequence[str] = ("object", "background"),
    normalize: bool = True,
    fname: str = "confusion_matrix.png",
) -> str:
    cm = np.asarray(cm, dtype=np.float64)
    if normalize:
        row_sum = cm.sum(axis=1, keepdims=True)
        row_sum[row_sum == 0] = 1.0
        cm_disp = cm / row_sum
    else:
        cm_disp = cm
    side = max(5.5, 1.2 * len(classes) + 3.0)
    fig, ax = plt.subplots(figsize=(side, side))
    vmax = 1 if normalize else cm_disp.max()
    im = ax.imshow(cm_disp, cmap="Blues", vmin=0, vmax=vmax)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    subject = f"confusion matrix ({('normalised' if normalize else 'counts')})"
    ax.set_title(_compose_title(method, subject, config), fontsize=11)
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=30, ha="right")
    ax.set_yticklabels(classes)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Ground truth")
    thresh = cm_disp.max() / 2.0
    for i in range(cm_disp.shape[0]):
        for j in range(cm_disp.shape[1]):
            val = cm_disp[i, j]
            txt = f"{val:.2f}" if normalize else f"{int(val):d}"
            ax.text(
                j,
                i,
                txt,
                ha="center",
                va="center",
                color="white" if val > thresh else "black",
                fontsize=11,
            )
    return _save(fig, os.path.join(out_dir, fname))

def _pr_from_scores(
    scores: np.ndarray, matched: np.ndarray, n_gt: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    scores = np.asarray(scores, dtype=np.float64)
    matched = np.asarray(matched, dtype=np.int64)
    n_tp = int(matched.sum())
    if len(scores) == 0 or n_gt == 0 or n_tp == 0:
        zero = np.array([0.0, 0.0])
        one = np.array([1.0, 0.0])
        return (one, zero, zero, 0.0)
    from sklearn.metrics import precision_recall_curve, average_precision_score

    k = n_tp / float(n_gt)
    precision, recall_raw, _ = precision_recall_curve(matched, scores)
    ap = float(average_precision_score(matched, scores)) * k
    precision = precision[::-1]
    recall = recall_raw[::-1] * k
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-09)
    return (precision, recall, f1, ap)

def plot_pr_curve(
    scores: np.ndarray,
    matched: np.ndarray,
    n_gt: int,
    out_dir: str,
    method: str = "RAILIGHT",
    config: Config = None,
    fname: str = "pr_curve.png",
) -> Tuple[str, float]:
    precision, recall, _, ap = _pr_from_scores(
        np.asarray(scores), np.asarray(matched), n_gt
    )
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(recall, precision, color=_BLUE, linewidth=1.6, label=f"AP={ap:.3f}")
    ax.fill_between(recall, precision, alpha=0.15, color=_BLUE)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(_compose_title(method, "Precision-Recall curve", config), fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="lower left")
    path = _save(fig, os.path.join(out_dir, fname))
    return (path, ap)

def plot_recall_f1_curve(
    scores: np.ndarray,
    matched: np.ndarray,
    n_gt: int,
    out_dir: str,
    method: str = "RAILIGHT",
    config: Config = None,
    fname: str = "recall_f1.png",
) -> str:
    _, recall, f1, _ = _pr_from_scores(np.asarray(scores), np.asarray(matched), n_gt)
    best_f1 = float(f1.max()) if len(f1) else 0.0
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(recall, f1, color=_BLUE, linewidth=1.6, label=f"best F1={best_f1:.3f}")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Recall")
    ax.set_ylabel("F1 score")
    ax.set_title(_compose_title(method, "Recall-F1 curve", config), fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="lower left")
    return _save(fig, os.path.join(out_dir, fname))

def _hist_panel(ax, series, colors, labels, title, xlabel, bins=40):
    """One publication-style distribution panel: filled hist + mean line."""
    for vals, col, lab in zip(series, colors, labels):
        v = np.asarray(vals, dtype=np.float64)
        v = v[np.isfinite(v)]
        if v.size == 0:
            continue
        ax.hist(
            v, bins=bins, density=True, alpha=0.45, color=col,
            edgecolor="white", linewidth=0.4,
            label=f"{lab}  (μ={v.mean():.3g}, n={v.size})",
        )
        ax.axvline(
            float(v.mean()), color=col, linestyle="--", linewidth=2.2,
        )
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    ax.set_xlabel(xlabel, fontsize=11, fontweight="bold")
    ax.set_ylabel("Density", fontsize=11, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.45)
    ax.legend(loc="best", framealpha=0.92, fontsize=9)

def plot_test_distributions(
    feat_day,
    feat_night,
    kl_vals,
    ce_vals,
    out_dir: str,
    method: str = "RAILIGHT",
    config: Config = None,
    fname: str = "test_distributions.png",
    feat_before_day=None,
    feat_before_night=None,
) -> Optional[str]:
  
    feat_before_day = feat_before_day or []
    feat_before_night = feat_before_night or []
    have_ba = bool(len(feat_before_day) or len(feat_before_night))
    have_feat = bool(len(feat_day) or len(feat_night))
    have_kl = bool(len(kl_vals))
    have_ce = bool(len(ce_vals))
    panels = [p for p, ok in (
        ("ba", have_ba), ("feat", have_feat), ("kl", have_kl), ("ce", have_ce)
    ) if ok]
    if not panels:
        return None

    n = len(panels)
    ncols = 2 if n >= 4 else n
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(6.8 * ncols, 5.0 * nrows), squeeze=False
    )
    axes = axes.flatten()
    for ax in axes[n:]:
        ax.axis("off")
    i = 0
    if have_ba:
        _hist_panel(
            axes[i],
            [feat_before_day, feat_before_night],
            [_BLUE, _RED],
            ["Day (source-val)", "Night (target)"],
            "Feature Distribution — Day vs Night\n"
            "(BEFORE DSFD.extract_features, raw-input mean activation)",
            "Mean input activation",
        )
        i += 1
    if have_feat:
        _hist_panel(
            axes[i],
            [feat_day, feat_night],
            [_BLUE, _RED],
            ["Day (source-val)", "Night (target)"],
            "Backbone-Feature Distribution — Day vs Night\n"
            "(AFTER DSFD.extract_features, per-image mean activation)",
            "Mean pooled-feature activation",
        )
        i += 1
    if have_kl:
        _hist_panel(
            axes[i],
            [kl_vals],
            ["#CC79A7"],
            ["Source ↔ Target"],
            "Domain KL-Divergence Distribution\n"
            "(source-val vs target backbone features)",
            "Per-batch symmetric KL  (lower = better aligned)",
        )
        i += 1
    if have_ce:
        _hist_panel(
            axes[i],
            [ce_vals],
            ["#0072B2"],
            ["H(day, night)"],
            "Cross-Entropy Distribution\n"
            "H(p_day, p_night)  (day↔night class distributions)",
            "Per-pair cross-entropy  (lower = more aligned)",
        )
        i += 1

    fig.suptitle(
        _compose_title(
            method, "Test-time Feature / KL / Cross-Entropy Distributions",
            config,
        ),
        fontsize=15, fontweight="bold", y=1.05,
    )
    fig.subplots_adjust(top=0.84)
    return _save(fig, os.path.join(out_dir, fname))

def _project_2d(feats: np.ndarray) -> Tuple[np.ndarray, str]:
    feats = np.asarray(feats, dtype=np.float64)
    if feats.shape[0] < 3:
        return (
            feats[:, :2] if feats.shape[1] >= 2 else np.zeros((feats.shape[0], 2)),
            "raw",
        )
    try:
        from sklearn.manifold import TSNE

        perp = max(5, min(30, (feats.shape[0] - 1) // 3))
        xy = TSNE(
            n_components=2, perplexity=perp, init="pca", learning_rate="auto"
        ).fit_transform(feats)
        return (xy, "t-SNE")
    except Exception:
        x = feats - feats.mean(axis=0, keepdims=True)
        try:
            _, _, vt = np.linalg.svd(x, full_matrices=False)
            return (x @ vt[:2].T, "PCA")
        except Exception:
            return (x[:, :2], "PCA(raw)")

def plot_tsne_features(
    feats: np.ndarray,
    labels: Sequence[int],
    class_names: Sequence[str],
    out_dir: str,
    fname: str = "tsne_source_features.png",
    method: str = "RAILIGHT",
    config: Config = None,
) -> Optional[str]:
    feats = np.asarray(feats, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int64)
    if feats.size == 0 or feats.shape[0] == 0:
        return None
    xy, proj = _project_2d(feats)
    fig, ax = plt.subplots(figsize=(7, 6))
    uniq = sorted(set((int(v) for v in labels)))
    for idx, cls in enumerate(uniq):
        m = labels == cls
        name = class_names[cls - 1] if 1 <= cls <= len(class_names) else f"class_{cls}"
        ax.scatter(
            xy[m, 0],
            xy[m, 1],
            s=18,
            alpha=0.7,
            color=_TAB_COLORS[idx % len(_TAB_COLORS)],
            label=f"{name} (n={int(m.sum())})",
        )
    ax.set_xlabel(f"{proj}-1")
    ax.set_ylabel(f"{proj}-2")
    ax.set_title(
        _compose_title(method, f"Source features ({proj}, by GT class)", config),
        fontsize=11,
    )
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="best", framealpha=0.9)
    return _save(fig, os.path.join(out_dir, fname))

def _subsample(a: np.ndarray, cap: int, seed: int) -> np.ndarray:
    if a is None or len(a) <= cap:
        return a
    idx = np.random.RandomState(seed).choice(len(a), size=cap, replace=False)
    return a[idx]

def _domain_scatter(ax, xy: np.ndarray, n_src: int, proj: str, title: str) -> None:
    # ECB-style domain t-SNE: source=red, target=blue, 'o' markers, alpha 0.5.
    ax.scatter(
        xy[:n_src, 0], xy[:n_src, 1],
        s=18, alpha=0.5, color="red", marker="o", label="Source Domain",
        edgecolors="none",
    )
    ax.scatter(
        xy[n_src:, 0], xy[n_src:, 1],
        s=18, alpha=0.5, color="blue", marker="o", label="Target Domain",
        edgecolors="none",
    )
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlabel(f"{proj}-1")
    ax.set_ylabel(f"{proj}-2")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(loc="upper right", framealpha=0.9, markerscale=1.6)

def plot_domain_tsne(
    feats_src: Optional[np.ndarray],
    feats_tgt: Optional[np.ndarray],
    out_dir: str,
    fname: str,
    subject: str,
    method: str = "RAILIGHT",
    config: Config = None,
    cap_per_domain: int = 1500,
) -> Optional[str]:
    if feats_src is None or feats_tgt is None:
        return None
    s = np.asarray(feats_src, dtype=np.float32).reshape(len(feats_src), -1)
    t = np.asarray(feats_tgt, dtype=np.float32).reshape(len(feats_tgt), -1)
    if s.shape[0] == 0 or t.shape[0] == 0:
        return None
    s = _subsample(s, cap_per_domain, seed=0)
    t = _subsample(t, cap_per_domain, seed=1)
    xy, proj = _project_2d(np.concatenate([s, t], axis=0))
    # ECB-style min-max scaling of the 2D embedding to [0, 1] per axis.
    mn = xy.min(axis=0, keepdims=True)
    mx = xy.max(axis=0, keepdims=True)
    xy = (xy - mn) / (mx - mn + 1e-9)
    fig, ax = plt.subplots(figsize=(10, 8))
    _domain_scatter(
        ax, xy, len(s), proj, _compose_title(method, subject, config)
    )
    return _save(fig, os.path.join(out_dir, fname))

def plot_domain_tsne_pair(
    panels: Sequence[Tuple[str, Optional[np.ndarray], Optional[np.ndarray]]],
    out_dir: str,
    fname: str,
    method: str = "RAILIGHT",
    config: Config = None,
    cap_per_domain: int = 1500,
    suptitle: Optional[str] = None,
) -> Optional[str]:
    prepared: List[Tuple[str, np.ndarray, int, Any]] = []
    for subject, feats_src, feats_tgt in panels:
        if feats_src is None or feats_tgt is None:
            continue
        s = np.asarray(feats_src, dtype=np.float32).reshape(len(feats_src), -1)
        t = np.asarray(feats_tgt, dtype=np.float32).reshape(len(feats_tgt), -1)
        if s.shape[0] == 0 or t.shape[0] == 0:
            continue
        s = _subsample(s, cap_per_domain, seed=0)
        t = _subsample(t, cap_per_domain, seed=1)
        xy, proj = _project_2d(np.concatenate([s, t], axis=0))
        mn = xy.min(axis=0, keepdims=True)
        mx = xy.max(axis=0, keepdims=True)
        prepared.append((subject, (xy - mn) / (mx - mn + 1e-9), len(s), proj))
    if not prepared:
        return None
    fig, axes = plt.subplots(
        1, len(prepared), figsize=(9.5 * len(prepared), 8), squeeze=False
    )
    for ax, (subject, xy, n_src, proj) in zip(axes[0], prepared):
        _domain_scatter(
            ax, xy, n_src, proj, _compose_title(method, subject, config)
        )
    if suptitle:
        fig.suptitle(suptitle, fontsize=14, fontweight="bold")
        fig.subplots_adjust(top=0.88)
    return _save(fig, os.path.join(out_dir, fname))

def plot_target_metrics(
    history: History,
    out_dir: str,
    method: str = "RAILIGHT",
    config: Config = None,
    fname: str = "target_metrics.png",
) -> Optional[str]:
    panels = [
        ("Loss", None, "target_val_loss"),
        ("Precision", None, "target_precision"),
        ("Recall", None, "target_recall"),
        ("F1 score", None, "target_f1"),
        ("mAP@0.5", None, "target_map"),
    ]
    panels = [
        p for p in panels
        if (p[1] and history.get(p[1])) or history.get(p[2])
    ]
    if not panels:
        return None
    n = len(panels)
    ncols = 2
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(7.5 * ncols, 4.2 * nrows), squeeze=False
    )
    for idx, (label, tr_key, va_key) in enumerate(panels):
        ax = axes[idx // ncols][idx % ncols]
        ax.set_facecolor("#ECECEC")
        if tr_key:
            tr = history.get(tr_key, [])
            if tr:
                xs, ys = zip(*tr)
                ax.plot(
                    xs, ys, color=_BLUE, linewidth=2.0, marker="o",
                    markersize=4, label="train (target sup)",
                )
        va = history.get(va_key, [])
        if va:
            xs, ys = zip(*va)
            ax.plot(
                xs, ys, color=_RED, linewidth=2.0, marker="s",
                markersize=4, label="val (target)",
            )
        ax.set_title(label, fontweight="bold", fontsize=12)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(label)
        ax.grid(True, linestyle="--", alpha=0.5, color="white", linewidth=1.0)
        ax.set_axisbelow(True)
        ax.legend(loc="best", framealpha=0.9)
    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].set_visible(False)
    run_tag = _run_tag(method, config)
    sup = "Target — train vs val metrics"
    if run_tag:
        sup = f"{sup}  ·  {run_tag}"
    fig.suptitle(sup, fontweight="bold", fontsize=13)
    fig.subplots_adjust(top=0.90)
    return _save(fig, os.path.join(out_dir, fname))

def plot_domain_metrics(
    history: History,
    out_dir: str,
    method: str = "RAILIGHT",
    config: Config = None,
    fname: str = "domain_metrics.png",
) -> Optional[str]:
    kl_pts = history.get("val_kl_st", [])
    ent_pts = history.get("val_entropy", [])
    mmd_pts = history.get("val_align_mmd", [])
    gap_pts = history.get("val_align_gap", [])
    if not kl_pts and not ent_pts and not mmd_pts and not gap_pts:
        return None
    fig, (ax1, ax3) = plt.subplots(1, 2, figsize=(15, 5))
    ax1.set_facecolor("#ECECEC")
    if kl_pts:
        xs, ys = zip(*kl_pts)
        ax1.plot(
            xs,
            ys,
            color=_BLUE,
            linewidth=2.0,
            marker="o",
            markersize=5,
            label="val alignment loss (source↔target)",
        )
    ax1.set_xlabel("Epoch", fontweight="bold")
    ax1.set_ylabel("Alignment loss", color=_BLUE, fontweight="bold")
    ax1.tick_params(axis="y", labelcolor=_BLUE)
    ax1.grid(True, linestyle="--", alpha=0.5, color="white", linewidth=1.2)
    ax1.set_axisbelow(True)
    if ent_pts:
        ax2 = ax1.twinx()
        xs, ys = zip(*ent_pts)
        ax2.plot(
            xs,
            ys,
            color=_RED,
            linewidth=2.0,
            marker="s",
            markersize=5,
            label="val entropy (target)",
        )
        ax2.set_ylabel("Target detection entropy", color=_RED, fontweight="bold")
        ax2.tick_params(axis="y", labelcolor=_RED)
    ax1.set_title("Optimised objective", fontweight="bold", fontsize=12, pad=10)
    lines, labs = ax1.get_legend_handles_labels()
    if ent_pts:
        l2, lb2 = ax2.get_legend_handles_labels()
        lines += l2
        labs += lb2
    ax1.legend(lines, labs, loc="best", framealpha=0.9)

    ax3.set_facecolor("#ECECEC")
    ax3.grid(True, linestyle="--", alpha=0.5, color="white", linewidth=1.2)
    ax3.set_axisbelow(True)
    ax3.set_xlabel("Epoch", fontweight="bold")
    ax3.set_ylabel("Embedding gap (std units)", color=_BLUE, fontweight="bold")
    ax3.tick_params(axis="y", labelcolor=_BLUE)
    if gap_pts:
        xs, ys = zip(*gap_pts)
        ax3.plot(
            xs, ys, color=_BLUE, linewidth=2.0, marker="o", markersize=5,
            label="source↔target mean gap",
        )
        ax3.axhline(
            0.2, color="#009E73", linestyle="--", linewidth=1.8,
            label="overlap threshold (0.2)",
        )
    if mmd_pts:
        ax4 = ax3.twinx()
        xs, ys = zip(*mmd_pts)
        ax4.plot(
            xs, ys, color="#CC79A7", linewidth=2.0, marker="^", markersize=5,
            label="source↔target MMD",
        )
        ax4.set_ylabel("MMD", color="#CC79A7", fontweight="bold")
        ax4.tick_params(axis="y", labelcolor="#CC79A7")
    ax3.set_title(
        "Actual embedding overlap (what t-SNE shows)",
        fontweight="bold", fontsize=12, pad=10,
    )
    lines3, labs3 = ax3.get_legend_handles_labels()
    if mmd_pts:
        l4, lb4 = ax4.get_legend_handles_labels()
        lines3 += l4
        labs3 += lb4
    ax3.legend(lines3, labs3, loc="best", framealpha=0.9)

    run_tag = _run_tag(method, config)
    suptitle = "Domain adaptation metrics"
    if run_tag:
        suptitle = f"{suptitle}  ·  {run_tag}"
    fig.suptitle(suptitle, fontweight="bold", fontsize=13)
    fig.subplots_adjust(top=0.86)
    return _save(fig, os.path.join(out_dir, fname))

def _draw_boxes(ax: "plt.Axes", sample: Dict[str, Any]) -> None:
    boxes = np.asarray(sample.get("boxes", []))
    scores = np.asarray(sample.get("scores", []))
    labels = list(sample.get("labels", [])) or ["face"] * len(boxes)
    for i, (x1, y1, x2, y2) in enumerate(boxes):
        ax.add_patch(
            Rectangle(
                (x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor="lime", linewidth=1.6
            )
        )
        ax.text(
            x1,
            max(0, y1 - 4),
            f"{labels[i]}:{scores[i]:.2f}",
            color="black",
            fontsize=8,
            bbox=dict(facecolor="lime", alpha=0.7, pad=1),
        )

def plot_sample_predictions(
    samples: Sequence[Dict[str, Any]],
    out_dir: str,
    fname: str,
    method: str = "RAILIGHT",
    config: Config = None,
    title_suffix: str = "",
) -> Optional[str]:
    n = len(samples)
    if n == 0:
        return None
    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(5.2 * ncols, 4.2 * nrows), squeeze=False
    )
    subject = f"sample predictions {title_suffix}".strip()
    fig.suptitle(_compose_title(method, subject, config), fontsize=12)
    for k, s in enumerate(samples):
        ax = axes[k // ncols][k % ncols]
        ax.imshow(s["image"])
        _draw_boxes(ax, s)
        ax.set_title(s.get("title", ""), fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
    for k in range(n, nrows * ncols):
        axes[k // ncols][k % ncols].set_visible(False)
    fig.subplots_adjust(top=0.9)
    return _save(fig, os.path.join(out_dir, fname))

def make_gradcam(net_inner: Any, target_layer: Any = None) -> Optional["GradCAM"]:
    import torch.nn as nn

    try:
        if target_layer is None:
            if hasattr(net_inner, "backbone") and hasattr(net_inner.backbone, "proj"):
                target_layer = net_inner.backbone.proj[-1]
            elif hasattr(net_inner, "vgg") and len(net_inner.vgg):
                convs = [m for m in net_inner.vgg if isinstance(m, nn.Conv2d)]
                target_layer = convs[-1] if convs else None
        if target_layer is None:
            return None
        return GradCAM(net_inner, target_layer)
    except Exception as e:
        print(f"[viz] grad-cam unavailable ({e})")
        return None


def collect_grid_items(
    loader: Any,
    detect_fn: Callable[[Any], Sequence[Tuple[np.ndarray, np.ndarray, np.ndarray]]],
    class_names: Sequence[str],
    n_show: int = 6,
    gradcam: Optional["GradCAM"] = None,
) -> List[Dict[str, Any]]:

    def _score(o):
        conf = o[4] if isinstance(o, (tuple, list)) else o
        return conf[..., 1:].max()

    def _fwd(m, t):
        return m.test_forward(t)[0]

    items: List[Dict[str, Any]] = []
    try:
        it = iter(loader)
        while len(items) < n_show:
            try:
                images, targets, paths = next(it)
            except StopIteration:
                break
            images = images.cuda() / 255.0
            dets = detect_fn(images)
            h_, w_ = (images.shape[2], images.shape[3])
            for i in range(images.shape[0]):
                if len(items) >= n_show:
                    break
                raw = (
                    images[i].detach().cpu().numpy().transpose(1, 2, 0)[:, :, ::-1]
                    * 255
                ).clip(0, 255).astype(np.uint8)
                pb, ps, pl = dets[i]
                gt = (
                    targets[i].cpu().numpy()
                    if hasattr(targets[i], "cpu")
                    else np.asarray(targets[i])
                )
                if gt.size:
                    gt_px = gt[:, :4].astype(np.float32).copy()
                    gt_px[:, [0, 2]] *= w_
                    gt_px[:, [1, 3]] *= h_
                else:
                    gt_px = np.zeros((0, 4), dtype=np.float32)
                cam = None
                if gradcam is not None:
                    try:
                        x = images[i : i + 1].detach().clone().requires_grad_(True)
                        cam = gradcam(x, score_fn=_score, forward_fn=_fwd)
                    except Exception as e:
                        print(f"[viz] grad-cam failed on a sample ({e})")
                        gradcam = None
                items.append(
                    dict(
                        image=raw,
                        cam01=cam,
                        boxes=pb,
                        scores=ps,
                        labels=[class_names[c - 1] for c in pl],
                        gt_boxes=gt_px,
                        title=os.path.basename(paths[i]) if i < len(paths) else "",
                    )
                )
    except Exception as e:
        print(f"[viz] sample-grid collection failed: {e}")
    return items


def plot_samples_grid_3row(
    items: Sequence[Dict[str, Any]],
    out_dir: str,
    fname: str,
    method: str = "RAILIGHT",
    config: Config = None,
    title_suffix: str = "",
) -> Optional[str]:

    n = len(items)
    if n == 0:
        return None
    fig, axes = plt.subplots(3, n, figsize=(3.4 * n, 10.2), squeeze=False)
    subject = f"input / Grad-CAM / detections {title_suffix}".strip()
    fig.suptitle(_compose_title(method, subject, config), fontsize=12)
    row_titles = ["Input", "Grad-CAM", "Detections (pred=lime, GT=red)"]
    for c, it in enumerate(items):
        img = it["image"]
        axes[0][c].imshow(img)
        if it.get("title"):
            axes[0][c].set_title(it["title"], fontsize=8)
        cam = it.get("cam01")
        if cam is not None:
            overlay, _ = _overlay_heatmap(img, cam)
            axes[1][c].imshow(overlay)
        else:
            axes[1][c].imshow(img)
        axes[2][c].imshow(img)
        _draw_boxes(
            axes[2][c],
            {
                "boxes": it.get("boxes", []),
                "scores": it.get("scores", []),
                "labels": it.get("labels", []),
            },
        )
        for x1, y1, x2, y2 in np.asarray(
            it.get("gt_boxes", []), dtype=np.float32
        ).reshape(-1, 4):
            axes[2][c].add_patch(
                Rectangle(
                    (x1, y1), x2 - x1, y2 - y1,
                    fill=False, edgecolor="red", linewidth=1.8,
                )
            )
        for r in range(3):
            axes[r][c].set_xticks([])
            axes[r][c].set_yticks([])
    for r in range(3):
        axes[r][0].set_ylabel(row_titles[r], fontsize=11, fontweight="bold")
    fig.subplots_adjust(top=0.92, hspace=0.06, wspace=0.04)
    return _save(fig, os.path.join(out_dir, fname))

def _iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    inter_x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    inter_y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    inter_x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    inter_y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    iw = np.clip(inter_x2 - inter_x1, 0, None)
    ih = np.clip(inter_y2 - inter_y1, 0, None)
    inter = iw * ih
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return inter / np.maximum(union, 1e-09)

def evaluate_detections(
    per_image: Iterable[Dict[str, Any]],
    iou_thr: float = 0.5,
    score_thr_cm: float = 0.5,
    num_classes: int = 1,
) -> Tuple[np.ndarray, np.ndarray, int, np.ndarray]:
    nc = max(int(num_classes), 1)
    cm = np.zeros((nc + 1, nc + 1), dtype=np.int64)
    all_scores: List[float] = []
    all_matched: List[int] = []
    n_gt_total = 0
    for item in per_image:
        pb = np.asarray(item.get("pred_boxes", []), dtype=np.float32)
        ps = np.asarray(item.get("pred_scores", []), dtype=np.float32)
        gb = np.asarray(item.get("gt_boxes", []), dtype=np.float32)
        pl = np.asarray(
            item.get("pred_labels", np.ones(len(pb), dtype=np.int32)), dtype=np.int32
        )
        gl = np.asarray(
            item.get("gt_labels", np.ones(len(gb), dtype=np.int32)), dtype=np.int32
        )
        n_gt_total += len(gb)
        if len(pb) == 0 and len(gb) == 0:
            continue
        order = np.argsort(-ps) if len(ps) else np.array([], dtype=int)
        pb_o = pb[order] if len(pb) else pb
        ps_o = ps[order] if len(ps) else ps
        pl_o = pl[order] if len(pl) else pl
        gt_used = np.zeros(len(gb), dtype=bool)
        matched_pred = np.zeros(len(pb_o), dtype=np.int32)
        if len(pb_o) and len(gb):
            ious = _iou_matrix(pb_o, gb)
            for i in range(len(pb_o)):
                j = int(np.argmax(ious[i]))
                if ious[i, j] >= iou_thr and (not gt_used[j]):
                    gt_used[j] = True
                    matched_pred[i] = 1
        all_scores.extend(ps_o.tolist())
        all_matched.extend(matched_pred.tolist())
        keep_mask = ps_o >= score_thr_cm
        kept_gt_used = np.zeros(len(gb), dtype=bool)
        if np.any(keep_mask) and len(pb_o):
            kept_pb = pb_o[keep_mask]
            kept_pl = pl_o[keep_mask]
            ious_kept = (
                _iou_matrix(kept_pb, gb) if len(gb) else np.zeros((len(kept_pb), 0))
            )
            for i in range(len(kept_pb)):
                p_idx = int(np.clip(kept_pl[i] - 1, 0, nc - 1))
                if len(gb) and ious_kept[i].size:
                    j = int(np.argmax(ious_kept[i]))
                    if ious_kept[i, j] >= iou_thr and (not kept_gt_used[j]):
                        kept_gt_used[j] = True
                        g_idx = int(np.clip(gl[j] - 1, 0, nc - 1))
                        cm[g_idx, p_idx] += 1
                        continue
                cm[nc, p_idx] += 1
        for j in range(len(gb)):
            if not kept_gt_used[j]:
                g_idx = int(np.clip(gl[j] - 1, 0, nc - 1))
                cm[g_idx, nc] += 1
    return (
        np.asarray(all_scores, dtype=np.float32),
        np.asarray(all_matched, dtype=np.int32),
        n_gt_total,
        cm,
    )

class GradCAM:

    def __init__(self, model: Any, target_layer: Any) -> None:
        import torch

        self._torch = torch
        self.model = model
        self.target_layer = target_layer
        self.activations: Optional[Any] = None
        self.gradients: Optional[Any] = None
        self._fwd_handle = target_layer.register_forward_hook(self._fwd_hook)

    def _fwd_hook(self, module: Any, inp: Any, out: Any) -> None:
        self.activations = out.detach().clone()
        if getattr(out, "requires_grad", False):
            out.register_hook(self._grad_hook)

    def _grad_hook(self, grad: Any) -> None:
        self.gradients = grad.detach()

    def remove(self) -> None:
        h = getattr(self, "_fwd_handle", None)
        if h is not None:
            h.remove()
        self._fwd_handle = None

    def _eigen_2d(self, weighted: Any) -> Any:
        """ECB/EigenCAM-style projection of weighted activations onto their
        first principal component across channels (per image)."""
        torch = self._torch
        b, c, h, w = weighted.shape
        maps = []
        for i in range(b):
            m = weighted[i].reshape(c, h * w).t()  # (HW, C)
            m = m - m.mean(dim=0, keepdim=True)
            try:
                _u, _s, vh = torch.linalg.svd(m, full_matrices=False)
                proj = (m @ vh[0]).reshape(h, w)
            except Exception:
                proj = weighted[i].sum(dim=0)
            maps.append(proj)
        return torch.stack(maps).unsqueeze(1)  # (B, 1, H, W)

    def _compute(
        self,
        x: Any,
        score_fn: Callable[[Any], Any],
        forward_fn: Optional[Callable[[Any, Any], Any]],
        eigen_smooth: bool,
    ) -> np.ndarray:
        torch = self._torch
        import torch.nn.functional as F

        self.model.zero_grad(set_to_none=True)
        out = (forward_fn or (lambda m, t: m(t)))(self.model, x)
        score_fn(out).backward(retain_graph=False)
        grads = self.gradients
        acts = self.activations
        if grads is None or acts is None:
            raise RuntimeError(
                "Grad-CAM: no gradients captured at the target layer "
                "(layer output may not require grad on this path)."
            )
        if grads.dim() == 4:
            weights = grads.mean(dim=(2, 3), keepdim=True)
        else:
            weights = grads.mean(dim=list(range(2, grads.dim())), keepdim=True)
        weighted = weights * acts
        if eigen_smooth:
            cam = self._eigen_2d(weighted)
        else:
            cam = weighted.sum(dim=1, keepdim=True)
        cam = torch.relu(cam)
        cam = F.interpolate(
            cam, size=x.shape[-2:], mode="bilinear", align_corners=False
        )
        cam = cam[0, 0].detach().cpu().numpy()
        mn, mx = (float(cam.min()), float(cam.max()))
        return (cam - mn) / (mx - mn + 1e-09)

    def __call__(
        self,
        x: Any,
        score_fn: Callable[[Any], Any],
        forward_fn: Optional[Callable[[Any, Any], Any]] = None,
        eigen_smooth: bool = False,
        aug_smooth: bool = False,
    ) -> np.ndarray:
        """Compute a normalized Grad-CAM map.

        ECB-style options: ``eigen_smooth`` projects the weighted activations
        onto their first principal component; ``aug_smooth`` averages the CAM
        over the input and its horizontal flip (test-time augmentation).
        """
        if not aug_smooth:
            return self._compute(x, score_fn, forward_fn, eigen_smooth)
        torch = self._torch
        cams = [self._compute(x, score_fn, forward_fn, eigen_smooth)]
        x_flip = torch.flip(x.detach(), dims=[-1]).requires_grad_(True)
        cam_flip = self._compute(x_flip, score_fn, forward_fn, eigen_smooth)
        cams.append(np.ascontiguousarray(cam_flip[:, ::-1]))
        cam = np.mean(cams, axis=0)
        mn, mx = (float(cam.min()), float(cam.max()))
        return (cam - mn) / (mx - mn + 1e-09)

def _overlay_heatmap(
    rgb_uint8: np.ndarray, cam01: np.ndarray, alpha: float = 0.45
) -> Tuple[np.ndarray, np.ndarray]:
    import cv2

    h, w = rgb_uint8.shape[:2]
    heat = (cam01 * 255).clip(0, 255).astype(np.uint8)
    if heat.shape != (h, w):
        heat = cv2.resize(heat, (w, h), interpolation=cv2.INTER_LINEAR)
    color = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
    color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
    out = (alpha * color + (1 - alpha) * rgb_uint8).clip(0, 255).astype(np.uint8)
    return (out, color)

def plot_gradcam_comparison(
    model: Any,
    target_layer: Any,
    source_items: Sequence[Dict[str, Any]],
    target_items: Sequence[Dict[str, Any]],
    out_dir: str,
    fname: str = "gradcam_source_vs_target.png",
    method: str = "RAILIGHT",
    config: Config = None,
    score_fn: Optional[Callable[[Any], Any]] = None,
    forward_fn: Optional[Callable[[Any, Any], Any]] = None,
    layer_name: str = "",
    eigen_smooth: bool = True,
    aug_smooth: bool = True,
) -> Optional[str]:
    import torch

    if score_fn is None:
        score_fn = lambda out: out.max()
    rows = [("Source (day)", source_items), ("Target (night)", target_items)]
    n_per_row = max(len(source_items), len(target_items))
    if n_per_row == 0:
        return None
    nrows = 2
    ncols = n_per_row * 3
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(2.3 * ncols, 2.6 * nrows + 0.6), squeeze=False
    )
    subject = "Grad-CAM (source vs target)"
    if layer_name:
        subject += f", layer={layer_name}"
    fig.suptitle(_compose_title(method, subject, config), fontsize=12)
    extractor = GradCAM(model, target_layer)
    try:
        was_training = model.training
        model.eval()
        for r, (row_label, items) in enumerate(rows):
            for c in range(n_per_row):
                if c >= len(items):
                    for k in range(3):
                        axes[r][c * 3 + k].set_visible(False)
                    continue
                item = items[c]
                rgb = np.asarray(item["image"]).astype(np.uint8)
                x = item["tensor"]
                if not isinstance(x, torch.Tensor):
                    raise TypeError("items must contain 'tensor' as a torch.Tensor")
                x = x.detach().clone().requires_grad_(True)
                cam = extractor(
                    x, score_fn=score_fn, forward_fn=forward_fn,
                    eigen_smooth=eigen_smooth, aug_smooth=aug_smooth,
                )
                overlay, heat_rgb = _overlay_heatmap(rgb, cam)
                ax0 = axes[r][c * 3 + 0]
                ax1 = axes[r][c * 3 + 1]
                ax2 = axes[r][c * 3 + 2]
                for ax in (ax0, ax1, ax2):
                    ax.set_xticks([])
                    ax.set_yticks([])
                ax0.imshow(rgb)
                ax1.imshow(heat_rgb)
                ax2.imshow(overlay)
                if c == 0:
                    ax0.set_ylabel(row_label, fontsize=10)
                if r == 0:
                    ax0.set_title(f"{item.get('title', '')}\noriginal", fontsize=8)
                    ax1.set_title("Grad-CAM", fontsize=8)
                    ax2.set_title("overlay", fontsize=8)
        if was_training:
            model.train()
    finally:
        extractor.remove()
    fig.subplots_adjust(top=0.86)
    return _save(fig, os.path.join(out_dir, fname))

def _parse_main_args(argv: Optional[Sequence[str]] = None) -> Any:
    import argparse

    p = argparse.ArgumentParser(
        description="Standalone Grad-CAM (source day vs target night) for RAILIGHT"
    )
    p.add_argument("cmd", nargs="?", default="gradcam", choices=["gradcam"])
    p.add_argument("--weights", required=True, type=str)
    p.add_argument(
        "--architecture",
        default="railight",
        type=str,
        help="Detection architecture name (used in charts path).",
    )
    p.add_argument(
        "--model",
        default="vgg16",
        type=str,
        choices=[
            "vgg16", "vgg16_sppf", "yolo26n", "yolo26s",
            "dark", "dark_sppf", "vgg", "resnet50", "resnet101", "resnet152",
        ],
        help="Backbone as written in configs/ (legacy model names still accepted).",
    )
    p.add_argument("--num_exp", default="exp1", type=str)
    p.add_argument("--charts_dir", default="./charts", type=str)
    p.add_argument("--mode_name", default="test", type=str)
    p.add_argument(
        "--target_folder",
        default="/media/caotulab/303A225B3A221DFA/Nhan/data/images/target",
        type=str,
    )
    p.add_argument("--day_folder", default="", type=str)
    p.add_argument("--num_pairs", default=3, type=int)
    p.add_argument("--target_layer_idx", default=22, type=int)
    p.add_argument("--forward_end_idx", default=30, type=int)
    return p.parse_args(argv)

def _load_images_from_folder(
    folder: str, max_n: int, size: int
) -> List[Tuple[np.ndarray, str]]:
    import glob
    from PIL import Image

    exts = (".jpg", ".jpeg", ".png", ".bmp", ".JPG", ".JPEG", ".PNG", ".BMP")
    files = sorted(
        (f for f in glob.glob(os.path.join(folder, "*")) if f.endswith(exts))
    )
    out: List[Tuple[np.ndarray, str]] = []
    for f in files[:max_n]:
        img = Image.open(f).convert("RGB").resize((size, size), Image.BILINEAR)
        out.append((np.asarray(img, dtype=np.uint8), os.path.basename(f)))
    return out

def _load_day_from_wider(max_n: int, size: int) -> List[Tuple[np.ndarray, str]]:
    from PIL import Image

    try:
        from data.config import cfg as _cfg
    except Exception as e:
        print("[main] cannot import data.config:", e)
        return []
    list_file = _cfg.FACE.VAL_FILE
    if not os.path.isfile(list_file):
        print("[main] WIDER val file not found:", list_file)
        return []
    out: List[Tuple[np.ndarray, str]] = []
    with open(list_file) as fh:
        for line in fh:
            parts = line.strip().split()
            if not parts:
                continue
            path = parts[0]
            if not os.path.isfile(path):
                continue
            try:
                img = (
                    Image.open(path).convert("RGB").resize((size, size), Image.BILINEAR)
                )
            except Exception:
                continue
            out.append((np.asarray(img, dtype=np.uint8), os.path.basename(path)))
            if len(out) >= max_n:
                break
    return out

def main(argv: Optional[Sequence[str]] = None) -> None:
    args = _parse_main_args(argv)
    import torch
    import torch.backends.cudnn as cudnn
    from data.config import cfg as dcfg
    from models.factory import build_net

    use_cuda = torch.cuda.is_available()
    if use_cuda:
        torch.set_default_tensor_type("torch.cuda.FloatTensor")
        cudnn.benchmark = True
    print(f"[main] building network ({args.model})")
    net = build_net(
        "test",
        num_classes=dcfg.NUM_CLASSES,
        backbone=args.model,
        architecture=args.architecture,
    )
    state = torch.load(args.weights, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "weight" in state:
        state = state["weight"]
    net.load_state_dict(state)
    net.eval()
    if use_cuda:
        net = net.cuda()
    if not hasattr(net, "vgg"):
        raise RuntimeError(
            "Backbone has no `.vgg` ModuleList; this Grad-CAM helper is wired for VGG-style RAILIGHT only."
        )
    target_idx = min(args.target_layer_idx, len(net.vgg) - 1)
    end = min(args.forward_end_idx, len(net.vgg))
    target_layer = net.vgg[target_idx]

    def forward_fn(_m: Any, x: Any) -> Any:
        h = x
        for k in range(end):
            h = net.vgg[k](h)
        return h

    def score_fn(out: Any) -> Any:
        return out.pow(2).mean()

    size = dcfg.INPUT_SIZE
    n = max(1, args.num_pairs)
    if args.day_folder and os.path.isdir(args.day_folder):
        day_imgs = _load_images_from_folder(args.day_folder, n, size)
        print(f"[main] day from folder {args.day_folder}: {len(day_imgs)} images")
    else:
        day_imgs = _load_day_from_wider(n, size)
        print(f"[main] day from WIDER val: {len(day_imgs)} images")
    if not os.path.isdir(args.target_folder):
        raise FileNotFoundError(f"target_folder does not exist: {args.target_folder}")
    night_imgs = _load_images_from_folder(args.target_folder, n, size)
    print(f"[main] night from {args.target_folder}: {len(night_imgs)} images")
    if not day_imgs or not night_imgs:
        raise RuntimeError("Need at least one source and one target image.")

    def to_item(pair: Tuple[np.ndarray, str]) -> Dict[str, Any]:
        rgb, name = pair
        tensor = (
            torch.from_numpy(rgb.astype(np.float32) / 255.0)
            .permute(2, 0, 1)
            .unsqueeze(0)
        )
        if use_cuda:
            tensor = tensor.cuda()
        return dict(image=rgb, tensor=tensor, title=name[:24])

    src_items = [to_item(p) for p in day_imgs[:n]]
    tgt_items = [to_item(p) for p in night_imgs[:n]]
    out_dir = make_charts_dir(
        args.charts_dir, args.mode_name, args.architecture, args.model, args.num_exp
    )
    method = f"RAILIGHT ({args.model}, {os.path.basename(args.weights)})"
    config = {
        "backbone": args.model,
        "weights": os.path.basename(args.weights),
        "pairs": n,
        "layer": f"vgg[{target_idx}]",
        "fwd_end": f"vgg[{end - 1}]",
    }
    path = plot_gradcam_comparison(
        net,
        target_layer,
        src_items,
        tgt_items,
        out_dir=out_dir,
        fname="gradcam_source_vs_target.png",
        method=method,
        config=config,
        score_fn=score_fn,
        forward_fn=forward_fn,
        layer_name=f"vgg[{target_idx}]->vgg[{end - 1}] (feature energy)",
    )
    print(f"[main] saved: {path}")

if __name__ == "__main__":
    main()
