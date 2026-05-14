"""Visualisation helpers for DAI-Net experiments.

Outputs are written to ``<charts_root>/<mode>/<backbone>/<num_exp>/``.

Produced charts (chart titles embed method + config so each figure stands alone):
    - losses.png                       subplots of every tracked loss curve
    - train_vs_val.png                 overlay of mean-train-loss vs val-loss
    - confusion_matrix.png             normalised, Blues colormap
    - pr_curve.png                     precision-recall curve + AP
    - recall_f1.png                    F1 score against recall
    - samples_*.png                    sample predictions with bounding boxes
    - gradcam_source_vs_target.png     source/target Grad-CAM comparison
"""
from __future__ import annotations

import math
import os
from typing import (
    Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union,
)

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
import numpy as np


matplotlib.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'legend.framealpha': 0.8,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
    'figure.facecolor': 'white',
    'axes.facecolor': '#F9F9F9',
})


Point = Tuple[float, float]
History = Dict[str, List[Point]]
Config = Optional[Dict[str, Any]]


_DPI = 200
_BLUE = 'tab:blue'
_RED = 'tab:red'
_TAB_COLORS = list(plt.get_cmap('tab10').colors)


_PRETTY_LOSS_NAME: Dict[str, str] = {
    'total': 'Loss',
    'pal1_loc': 'Pal1 Loc',
    'pal1_conf': 'Pal1 Conf',
    'pal2_loc': 'Pal2 Loc',
    'pal2_conf': 'Pal2 Conf',
    'enhance': 'Enhance',
    'enhance_l1ssim': 'Enhance L1+SSIM',
    'mutual': 'Mutual',
    'train_loss_epoch': 'Train Loss (epoch)',
    'val_loss': 'Val Loss',
}

_LOSS_ORDER: Tuple[str, ...] = (
    'total', 'pal1_loc', 'pal1_conf', 'pal2_loc', 'pal2_conf',
    'enhance', 'enhance_l1ssim', 'mutual',
)
_EPOCH_KEYS = frozenset({'train_loss_epoch', 'val_loss'})


def _run_tag(method: str, config: Config) -> str:
    """Compact identifier used in chart titles (e.g. "DAI-Net | (dark) | Exp exp1")."""
    parts: List[str] = []
    if method:
        parts.append(str(method))
    if config:
        bb = config.get('backbone')
        if bb:
            parts.append(f'({bb})')
        exp = config.get('exp') or config.get('num_exp')
        if exp:
            parts.append(f'Exp {exp}')
    return ' | '.join(parts)


def make_charts_dir(charts_root: str, mode: str,
                    architecture: str, backbone: str,
                    num_exp: str) -> str:
    out = os.path.join(
        str(charts_root), str(mode),
        str(architecture), str(backbone), str(num_exp),
    )
    os.makedirs(out, exist_ok=True)
    return out


def _config_suffix(config: Config) -> str:
    if not config:
        return ''
    return ' | '.join(f'{k}={v}' for k, v in config.items())


def _compose_title(method: str, subject: str, config: Config) -> str:
    title = f'{method} — {subject}'
    sub = _config_suffix(config)
    if sub:
        title += f'\n({sub})'
    return title


def _save(fig: Figure, path: str) -> str:
    fig.tight_layout()
    fig.savefig(path, dpi=_DPI, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    return path


def plot_losses(history: History, out_dir: str,
                method: str = 'DAI-Net',
                config: Config = None,
                x_key: str = 'iter') -> Optional[str]:
    """Polished loss grid — one subplot per metric, val_loss / train epoch last."""
    if not history:
        return None

    # Keep only series with data, in a stable & meaningful order.
    iter_keys = [k for k in _LOSS_ORDER if k in history and history[k]]
    extra_iter = sorted(
        k for k, v in history.items()
        if v and k not in iter_keys and k not in _EPOCH_KEYS
    )
    epoch_keys = [k for k in ('train_loss_epoch', 'val_loss')
                  if k in history and history[k]]
    ordered = iter_keys + extra_iter + epoch_keys
    if not ordered:
        return None

    n = len(ordered)
    ncols = min(4, n)
    nrows = math.ceil(n / ncols)
    run_tag = _run_tag(method, config)
    # Wrap long run_tag onto multiple lines inside each subplot title so it
    # never overflows horizontally into the neighbouring subplot.
    run_tag_lines = run_tag.split(' | ') if run_tag else []

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(6 * ncols, 5 * nrows),
        squeeze=False,
    )
    axes_flat = axes.flatten()

    for i, key in enumerate(ordered):
        ax = axes_flat[i]
        pts = history[key]
        xs, ys = zip(*pts)
        color = _TAB_COLORS[i % len(_TAB_COLORS)]
        is_epoch = key in _EPOCH_KEYS
        marker = 's' if is_epoch else 'o'
        markersize = 4 if is_epoch else 3
        ax.plot(xs, ys, color=color, linewidth=2.0,
                marker=marker, markersize=markersize)
        short = _PRETTY_LOSS_NAME.get(key, key.replace('_', ' ').title())
        title_lines = [short] + run_tag_lines
        ax.set_title('\n'.join(title_lines),
                     fontweight='bold', fontsize=11, pad=10)
        ax.set_xlabel('Epoch' if is_epoch else 'Iteration', fontweight='bold')
        ylabel = 'Loss' if key == 'val_loss' else 'Value'
        ax.set_ylabel(ylabel, fontweight='bold')

    for i in range(n, len(axes_flat)):
        axes_flat[i].set_visible(False)

    if run_tag:
        fig.suptitle(f'Loss Components — {run_tag}',
                     fontsize=14, fontweight='bold', y=1.005)
    fig.tight_layout(h_pad=2.5, w_pad=1.5, rect=(0, 0, 1, 0.985))
    fig.savefig(os.path.join(out_dir, 'losses.png'),
                dpi=_DPI, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    return os.path.join(out_dir, 'losses.png')


def plot_train_vs_val(train_pts: Sequence[Point],
                      val_pts: Sequence[Point],
                      out_dir: str,
                      method: str = 'DAI-Net',
                      config: Config = None,
                      fname: str = 'train_vs_val.png') -> Optional[str]:
    if not train_pts and not val_pts:
        return None
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_facecolor('#ECECEC')
    if train_pts:
        xs, ys = zip(*train_pts)
        ax.plot(xs, ys, color=_BLUE, linewidth=2.0, marker='o', markersize=5,
                label='train (pal2 det / epoch)')
    if val_pts:
        xs, ys = zip(*val_pts)
        ax.plot(xs, ys, color=_RED, linewidth=2.0, marker='s', markersize=5,
                label='val (target proxy_loss)')
    ax.set_xlabel('Epoch', fontweight='bold')
    ax.set_ylabel('Loss', fontweight='bold')
    run_tag = _run_tag(method, config)
    title = 'Train vs Val'
    if run_tag:
        title = f'{title}\n{run_tag}'
    ax.set_title(title, fontweight='bold', fontsize=13, pad=10)
    ax.grid(True, linestyle='--', alpha=0.5, color='white', linewidth=1.2)
    ax.set_axisbelow(True)
    ax.legend(loc='best', framealpha=0.9)
    return _save(fig, os.path.join(out_dir, fname))


def plot_confusion_matrix(cm: np.ndarray, out_dir: str,
                          method: str = 'DAI-Net',
                          config: Config = None,
                          classes: Sequence[str] = ('object', 'background'),
                          normalize: bool = True) -> str:
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
    im = ax.imshow(cm_disp, cmap='Blues', vmin=0, vmax=vmax)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    subject = (
        f'confusion matrix ({"normalised" if normalize else "counts"})'
    )
    ax.set_title(_compose_title(method, subject, config), fontsize=11)

    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=30, ha='right')
    ax.set_yticklabels(classes)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Ground truth')

    thresh = cm_disp.max() / 2.0
    for i in range(cm_disp.shape[0]):
        for j in range(cm_disp.shape[1]):
            val = cm_disp[i, j]
            txt = f'{val:.2f}' if normalize else f'{int(val):d}'
            ax.text(
                j, i, txt, ha='center', va='center',
                color='white' if val > thresh else 'black', fontsize=11,
            )

    return _save(fig, os.path.join(out_dir, 'confusion_matrix.png'))


def _pr_from_scores(
    scores: np.ndarray, matched: np.ndarray, n_gt: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    if len(scores) == 0 or n_gt == 0:
        zero = np.array([0.0, 0.0])
        one = np.array([1.0, 0.0])
        return one, zero, zero, 0.0
    order = np.argsort(-scores)
    matched = matched[order]
    tp = np.cumsum(matched)
    fp = np.cumsum(1 - matched)
    recall = tp / float(n_gt)
    precision = tp / np.maximum(tp + fp, 1e-9)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-9)

    ap = 0.0
    for t in np.linspace(0, 1, 11):
        p = precision[recall >= t].max() if np.any(recall >= t) else 0.0
        ap += p / 11.0
    return precision, recall, f1, ap


def plot_pr_curve(scores: np.ndarray, matched: np.ndarray, n_gt: int,
                  out_dir: str,
                  method: str = 'DAI-Net',
                  config: Config = None) -> Tuple[str, float]:
    precision, recall, _, ap = _pr_from_scores(
        np.asarray(scores), np.asarray(matched), n_gt,
    )
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(recall, precision, color=_BLUE, linewidth=1.6,
            label=f'AP={ap:.3f}')
    ax.fill_between(recall, precision, alpha=0.15, color=_BLUE)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title(_compose_title(method, 'Precision-Recall curve', config),
                 fontsize=11)
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.legend(loc='lower left')
    path = _save(fig, os.path.join(out_dir, 'pr_curve.png'))
    return path, ap


def plot_recall_f1_curve(scores: np.ndarray, matched: np.ndarray, n_gt: int,
                         out_dir: str,
                         method: str = 'DAI-Net',
                         config: Config = None) -> str:
    _, recall, f1, _ = _pr_from_scores(
        np.asarray(scores), np.asarray(matched), n_gt,
    )
    best_f1 = float(f1.max()) if len(f1) else 0.0
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(recall, f1, color=_BLUE, linewidth=1.6,
            label=f'best F1={best_f1:.3f}')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel('Recall')
    ax.set_ylabel('F1 score')
    ax.set_title(_compose_title(method, 'Recall-F1 curve', config), fontsize=11)
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.legend(loc='lower left')
    return _save(fig, os.path.join(out_dir, 'recall_f1.png'))


def _draw_boxes(ax: 'plt.Axes', sample: Dict[str, Any]) -> None:
    boxes = np.asarray(sample.get('boxes', []))
    scores = np.asarray(sample.get('scores', []))
    labels = list(sample.get('labels', [])) or ['face'] * len(boxes)
    for i, (x1, y1, x2, y2) in enumerate(boxes):
        ax.add_patch(Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            fill=False, edgecolor='lime', linewidth=1.6,
        ))
        ax.text(
            x1, max(0, y1 - 4),
            f'{labels[i]}:{scores[i]:.2f}',
            color='black', fontsize=8,
            bbox=dict(facecolor='lime', alpha=0.7, pad=1),
        )


def plot_sample_predictions(samples: Sequence[Dict[str, Any]], out_dir: str,
                            fname: str,
                            method: str = 'DAI-Net',
                            config: Config = None,
                            title_suffix: str = '') -> Optional[str]:
    n = len(samples)
    if n == 0:
        return None
    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(5.2 * ncols, 4.2 * nrows), squeeze=False,
    )

    subject = f'sample predictions {title_suffix}'.strip()
    fig.suptitle(_compose_title(method, subject, config), fontsize=12)

    for k, s in enumerate(samples):
        ax = axes[k // ncols][k % ncols]
        ax.imshow(s['image'])
        _draw_boxes(ax, s)
        ax.set_title(s.get('title', ''), fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
    for k in range(n, nrows * ncols):
        axes[k // ncols][k % ncols].set_visible(False)

    fig.subplots_adjust(top=0.9)
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
    return inter / np.maximum(union, 1e-9)


def evaluate_detections(per_image: Iterable[Dict[str, Any]],
                        iou_thr: float = 0.5,
                        score_thr_cm: float = 0.5,
                        num_classes: int = 1,
                        ) -> Tuple[np.ndarray, np.ndarray, int, np.ndarray]:
    """Match predictions to GT for PR / confusion-matrix charts.

    Each `per_image` item has keys: 'pred_boxes', 'pred_scores', 'gt_boxes',
    and optionally 'pred_labels' / 'gt_labels' (1-indexed; 1..num_classes).

    Returns (scores, matched, n_gt, cm). The CM has shape
    (num_classes+1, num_classes+1) where row/col i (0..nc-1) is foreground
    class i and row/col nc is background.
        - cm[gt, pred]: GT of class `gt` predicted as class `pred`
        - cm[gt, nc]:   missed detection (FN) for class `gt`
        - cm[nc, pred]: false positive of class `pred` (no matching GT)
    """
    nc = max(int(num_classes), 1)
    cm = np.zeros((nc + 1, nc + 1), dtype=np.int64)
    all_scores: List[float] = []
    all_matched: List[int] = []
    n_gt_total = 0

    for item in per_image:
        pb = np.asarray(item.get('pred_boxes', []), dtype=np.float32)
        ps = np.asarray(item.get('pred_scores', []), dtype=np.float32)
        gb = np.asarray(item.get('gt_boxes', []), dtype=np.float32)
        pl = np.asarray(item.get('pred_labels',
                                 np.ones(len(pb), dtype=np.int32)),
                        dtype=np.int32)
        gl = np.asarray(item.get('gt_labels',
                                 np.ones(len(gb), dtype=np.int32)),
                        dtype=np.int32)
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
                if ious[i, j] >= iou_thr and not gt_used[j]:
                    gt_used[j] = True
                    matched_pred[i] = 1

        all_scores.extend(ps_o.tolist())
        all_matched.extend(matched_pred.tolist())

        # Build CM from kept (above-threshold) predictions only.
        keep_mask = ps_o >= score_thr_cm
        kept_gt_used = np.zeros(len(gb), dtype=bool)
        if np.any(keep_mask) and len(pb_o):
            kept_pb = pb_o[keep_mask]
            kept_pl = pl_o[keep_mask]
            ious_kept = (_iou_matrix(kept_pb, gb)
                         if len(gb) else np.zeros((len(kept_pb), 0)))
            for i in range(len(kept_pb)):
                p_idx = int(np.clip(kept_pl[i] - 1, 0, nc - 1))
                if len(gb) and ious_kept[i].size:
                    j = int(np.argmax(ious_kept[i]))
                    if ious_kept[i, j] >= iou_thr and not kept_gt_used[j]:
                        kept_gt_used[j] = True
                        g_idx = int(np.clip(gl[j] - 1, 0, nc - 1))
                        cm[g_idx, p_idx] += 1
                        continue
                cm[nc, p_idx] += 1  # FP — no matching GT
        # Any GT not matched by a kept prediction is a missed detection.
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
    """Grad-CAM extractor that hooks a single target module.

    Usage:
        cam = GradCAM(model, target_layer)
        heat = cam(x, score_fn=lambda out: out[..., cls].max())
        cam.remove()
    """

    def __init__(self, model: Any, target_layer: Any) -> None:
        import torch  # local import — torch is heavy
        self._torch = torch
        self.model = model
        self.target_layer = target_layer
        self.activations: Optional[Any] = None
        self.gradients: Optional[Any] = None
        self._fwd_handle = target_layer.register_forward_hook(self._fwd_hook)
        if hasattr(target_layer, 'register_full_backward_hook'):
            self._bwd_handle = target_layer.register_full_backward_hook(
                self._bwd_hook,
            )
        else:
            self._bwd_handle = target_layer.register_backward_hook(
                self._bwd_hook,
            )

    def _fwd_hook(self, module: Any, inp: Any, out: Any) -> None:
        self.activations = out.detach() if hasattr(out, 'detach') else out

    def _bwd_hook(self, module: Any, grad_in: Any, grad_out: Any) -> None:
        self.gradients = grad_out[0].detach()

    def remove(self) -> None:
        for h in (getattr(self, '_fwd_handle', None),
                  getattr(self, '_bwd_handle', None)):
            if h is not None:
                h.remove()
        self._fwd_handle = None
        self._bwd_handle = None

    def __call__(self,
                 x: Any,
                 score_fn: Callable[[Any], Any],
                 forward_fn: Optional[Callable[[Any, Any], Any]] = None,
                 ) -> np.ndarray:
        torch = self._torch
        import torch.nn.functional as F

        self.model.zero_grad(set_to_none=True)
        out = (forward_fn or (lambda m, t: m(t)))(self.model, x)
        score_fn(out).backward(retain_graph=False)

        grads = self.gradients
        acts = self.activations
        if grads.dim() == 4:
            weights = grads.mean(dim=(2, 3), keepdim=True)
        else:
            weights = grads.mean(
                dim=list(range(2, grads.dim())), keepdim=True,
            )
        cam = (weights * acts).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)
        cam = F.interpolate(
            cam, size=x.shape[-2:], mode='bilinear', align_corners=False,
        )
        cam = cam[0, 0].detach().cpu().numpy()
        mn, mx = float(cam.min()), float(cam.max())
        return (cam - mn) / (mx - mn + 1e-9)


def _overlay_heatmap(rgb_uint8: np.ndarray, cam01: np.ndarray,
                     alpha: float = 0.45) -> Tuple[np.ndarray, np.ndarray]:
    import cv2  # local import — cv2 only needed for the overlay

    h, w = rgb_uint8.shape[:2]
    heat = (cam01 * 255).clip(0, 255).astype(np.uint8)
    if heat.shape != (h, w):
        heat = cv2.resize(heat, (w, h), interpolation=cv2.INTER_LINEAR)
    color = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
    color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
    out = (alpha * color + (1 - alpha) * rgb_uint8).clip(0, 255).astype(np.uint8)
    return out, color


def plot_gradcam_comparison(model: Any, target_layer: Any,
                            source_items: Sequence[Dict[str, Any]],
                            target_items: Sequence[Dict[str, Any]],
                            out_dir: str,
                            fname: str = 'gradcam_source_vs_target.png',
                            method: str = 'DAI-Net',
                            config: Config = None,
                            score_fn: Optional[Callable[[Any], Any]] = None,
                            forward_fn: Optional[
                                Callable[[Any, Any], Any]
                            ] = None,
                            layer_name: str = '') -> Optional[str]:
    """Side-by-side Grad-CAM panel for source (day) and target (night) images.

    Each item is a dict with keys:
        image:  HxWx3 uint8 RGB
        tensor: (1,3,H,W) torch tensor on the model's device
        title:  short caption (optional)
    """
    import torch

    if score_fn is None:
        score_fn = lambda out: out.max()  # noqa: E731

    rows = [('Source (day)', source_items), ('Target (night)', target_items)]
    n_per_row = max(len(source_items), len(target_items))
    if n_per_row == 0:
        return None

    nrows = 2
    ncols = n_per_row * 3
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(2.3 * ncols, 2.6 * nrows + 0.6),
        squeeze=False,
    )

    subject = 'Grad-CAM (source vs target)'
    if layer_name:
        subject += f', layer={layer_name}'
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
                rgb = np.asarray(item['image']).astype(np.uint8)
                x = item['tensor']
                if not isinstance(x, torch.Tensor):
                    raise TypeError(
                        "items must contain 'tensor' as a torch.Tensor"
                    )
                x = x.detach().clone().requires_grad_(True)
                cam = extractor(x, score_fn=score_fn, forward_fn=forward_fn)
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
                    ax0.set_title(
                        f'{item.get("title", "")}\noriginal', fontsize=8,
                    )
                    ax1.set_title('Grad-CAM', fontsize=8)
                    ax2.set_title('overlay', fontsize=8)
        if was_training:
            model.train()
    finally:
        extractor.remove()

    fig.subplots_adjust(top=0.86)
    return _save(fig, os.path.join(out_dir, fname))


def _parse_main_args(argv: Optional[Sequence[str]] = None) -> Any:
    import argparse
    p = argparse.ArgumentParser(
        description='Standalone Grad-CAM (source day vs target night) for DAI-Net',
    )
    p.add_argument('cmd', nargs='?', default='gradcam', choices=['gradcam'])
    p.add_argument('--weights', required=True, type=str)
    p.add_argument(
        '--architecture', default='dai_net', type=str,
        help='Detection architecture name (used in charts path).',
    )
    p.add_argument(
        '--model', default='dark', type=str,
        choices=['dark', 'vgg', 'resnet50', 'resnet101', 'resnet152'],
    )
    p.add_argument('--num_exp', default='exp1', type=str)
    p.add_argument('--charts_dir', default='./charts', type=str)
    p.add_argument('--mode_name', default='test', type=str)
    p.add_argument(
        '--target_folder',
        default='/media/caotulab/303A225B3A221DFA/Nhan/data/images/target',
        type=str,
    )
    p.add_argument('--day_folder', default='', type=str)
    p.add_argument('--num_pairs', default=3, type=int)
    p.add_argument('--target_layer_idx', default=22, type=int)
    p.add_argument('--forward_end_idx', default=30, type=int)
    return p.parse_args(argv)


def _load_images_from_folder(
    folder: str, max_n: int, size: int,
) -> List[Tuple[np.ndarray, str]]:
    import glob
    from PIL import Image
    exts = ('.jpg', '.jpeg', '.png', '.bmp', '.JPG', '.JPEG', '.PNG', '.BMP')
    files = sorted(
        f for f in glob.glob(os.path.join(folder, '*')) if f.endswith(exts)
    )
    out: List[Tuple[np.ndarray, str]] = []
    for f in files[:max_n]:
        img = (Image.open(f)
               .convert('RGB')
               .resize((size, size), Image.BILINEAR))
        out.append((np.asarray(img, dtype=np.uint8), os.path.basename(f)))
    return out


def _load_day_from_wider(max_n: int, size: int,
                         ) -> List[Tuple[np.ndarray, str]]:
    from PIL import Image
    try:
        from data.config import cfg as _cfg
    except Exception as e:
        print('[main] cannot import data.config:', e)
        return []

    list_file = _cfg.FACE.VAL_FILE
    if not os.path.isfile(list_file):
        print('[main] WIDER val file not found:', list_file)
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
                img = (Image.open(path)
                       .convert('RGB')
                       .resize((size, size), Image.BILINEAR))
            except Exception:
                continue
            out.append((np.asarray(img, dtype=np.uint8),
                        os.path.basename(path)))
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
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        cudnn.benchmark = True

    print(f'[main] building network ({args.model})')
    net = build_net('test', num_classes=dcfg.NUM_CLASSES, model=args.model)
    state = torch.load(args.weights, map_location='cpu')
    if isinstance(state, dict) and 'weight' in state:
        state = state['weight']
    net.load_state_dict(state)
    net.eval()
    if use_cuda:
        net = net.cuda()

    if not hasattr(net, 'vgg'):
        raise RuntimeError(
            'Backbone has no `.vgg` ModuleList; this Grad-CAM helper is '
            'wired for VGG-style DAI-Net only.'
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
        print(f'[main] day from folder {args.day_folder}: {len(day_imgs)} images')
    else:
        day_imgs = _load_day_from_wider(n, size)
        print(f'[main] day from WIDER val: {len(day_imgs)} images')

    if not os.path.isdir(args.target_folder):
        raise FileNotFoundError(
            f'target_folder does not exist: {args.target_folder}'
        )
    night_imgs = _load_images_from_folder(args.target_folder, n, size)
    print(f'[main] night from {args.target_folder}: {len(night_imgs)} images')

    if not day_imgs or not night_imgs:
        raise RuntimeError('Need at least one source and one target image.')

    def to_item(pair: Tuple[np.ndarray, str]) -> Dict[str, Any]:
        rgb, name = pair
        tensor = (torch.from_numpy(rgb.astype(np.float32) / 255.0)
                       .permute(2, 0, 1)
                       .unsqueeze(0))
        if use_cuda:
            tensor = tensor.cuda()
        return dict(image=rgb, tensor=tensor, title=name[:24])

    src_items = [to_item(p) for p in day_imgs[:n]]
    tgt_items = [to_item(p) for p in night_imgs[:n]]

    out_dir = make_charts_dir(
        args.charts_dir, args.mode_name,
        args.architecture, args.model, args.num_exp,
    )
    method = f'DAI-Net ({args.model}, {os.path.basename(args.weights)})'
    config = {
        'backbone': args.model,
        'weights': os.path.basename(args.weights),
        'pairs': n,
        'layer': f'vgg[{target_idx}]',
        'fwd_end': f'vgg[{end - 1}]',
    }

    path = plot_gradcam_comparison(
        net, target_layer, src_items, tgt_items,
        out_dir=out_dir,
        fname='gradcam_source_vs_target.png',
        method=method, config=config,
        score_fn=score_fn, forward_fn=forward_fn,
        layer_name=f'vgg[{target_idx}]->vgg[{end - 1}] (feature energy)',
    )
    print(f'[main] saved: {path}')


if __name__ == '__main__':
    main()
