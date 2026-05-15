from __future__ import annotations
import math
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any
import numpy as np
import torch
from ultralytics.utils import (
    LOGGER,
    DataExportMixin,
    SimpleClass,
    TryExcept,
    checks,
    plt_settings,
)

OKS_SIGMA = (
    np.array(
        [
            0.26,
            0.25,
            0.25,
            0.35,
            0.35,
            0.79,
            0.79,
            0.72,
            0.72,
            0.62,
            0.62,
            1.07,
            1.07,
            0.87,
            0.87,
            0.89,
            0.89,
        ],
        dtype=np.float32,
    )
    / 10.0
)
RLE_WEIGHT = np.array(
    [
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.2,
        1.2,
        1.5,
        1.5,
        1.0,
        1.0,
        1.2,
        1.2,
        1.5,
        1.5,
    ]
)


def bbox_ioa(
    box1: np.ndarray, box2: np.ndarray, iou: bool = False, eps: float = 1e-07
) -> np.ndarray:
    b1_x1, b1_y1, b1_x2, b1_y2 = box1.T
    b2_x1, b2_y1, b2_x2, b2_y2 = box2.T
    inter_area = (
        np.minimum(b1_x2[:, None], b2_x2) - np.maximum(b1_x1[:, None], b2_x1)
    ).clip(0) * (
        np.minimum(b1_y2[:, None], b2_y2) - np.maximum(b1_y1[:, None], b2_y1)
    ).clip(
        0
    )
    area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
    if iou:
        box1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
        area = area + box1_area[:, None] - inter_area
    return inter_area / (area + eps)


def box_iou(box1: torch.Tensor, box2: torch.Tensor, eps: float = 1e-07) -> torch.Tensor:
    (a1, a2), (b1, b2) = (
        box1.float().unsqueeze(1).chunk(2, 2),
        box2.float().unsqueeze(0).chunk(2, 2),
    )
    inter = (torch.min(a2, b2) - torch.max(a1, b1)).clamp_(0).prod(2)
    return inter / ((a2 - a1).prod(2) + (b2 - b1).prod(2) - inter + eps)


def bbox_iou(
    box1: torch.Tensor,
    box2: torch.Tensor,
    xywh: bool = True,
    GIoU: bool = False,
    DIoU: bool = False,
    CIoU: bool = False,
    eps: float = 1e-07,
) -> torch.Tensor:
    if xywh:
        (x1, y1, w1, h1), (x2, y2, w2, h2) = (box1.chunk(4, -1), box2.chunk(4, -1))
        w1_, h1_, w2_, h2_ = (w1 / 2, h1 / 2, w2 / 2, h2 / 2)
        b1_x1, b1_x2, b1_y1, b1_y2 = (x1 - w1_, x1 + w1_, y1 - h1_, y1 + h1_)
        b2_x1, b2_x2, b2_y1, b2_y2 = (x2 - w2_, x2 + w2_, y2 - h2_, y2 + h2_)
    else:
        b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)
        b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)
        w1, h1 = (b1_x2 - b1_x1, b1_y2 - b1_y1 + eps)
        w2, h2 = (b2_x2 - b2_x1, b2_y2 - b2_y1 + eps)
    inter = (b1_x2.minimum(b2_x2) - b1_x1.maximum(b2_x1)).clamp_(0) * (
        b1_y2.minimum(b2_y2) - b1_y1.maximum(b2_y1)
    ).clamp_(0)
    union = w1 * h1 + w2 * h2 - inter + eps
    iou = inter / union
    if CIoU or DIoU or GIoU:
        cw = b1_x2.maximum(b2_x2) - b1_x1.minimum(b2_x1)
        ch = b1_y2.maximum(b2_y2) - b1_y1.minimum(b2_y1)
        if CIoU or DIoU:
            c2 = cw.pow(2) + ch.pow(2) + eps
            rho2 = (
                (b2_x1 + b2_x2 - b1_x1 - b1_x2).pow(2)
                + (b2_y1 + b2_y2 - b1_y1 - b1_y2).pow(2)
            ) / 4
            if CIoU:
                v = 4 / math.pi**2 * ((w2 / h2).atan() - (w1 / h1).atan()).pow(2)
                with torch.no_grad():
                    alpha = v / (v - iou + (1 + eps))
                return iou - (rho2 / c2 + v * alpha)
            return iou - rho2 / c2
        c_area = cw * ch + eps
        return iou - (c_area - union) / c_area
    return iou


def mask_iou(
    mask1: torch.Tensor, mask2: torch.Tensor, eps: float = 1e-07
) -> torch.Tensor:
    intersection = torch.matmul(mask1, mask2.T).clamp_(0)
    union = mask1.sum(1)[:, None] + mask2.sum(1)[None] - intersection
    return intersection / (union + eps)


def kpt_iou(
    kpt1: torch.Tensor,
    kpt2: torch.Tensor,
    area: torch.Tensor,
    sigma: list[float],
    eps: float = 1e-07,
) -> torch.Tensor:
    d = (kpt1[:, None, :, 0] - kpt2[..., 0]).pow(2) + (
        kpt1[:, None, :, 1] - kpt2[..., 1]
    ).pow(2)
    sigma = torch.tensor(sigma, device=kpt1.device, dtype=kpt1.dtype)
    kpt_mask = kpt1[..., 2] != 0
    e = d / ((2 * sigma).pow(2) * (area[:, None, None] + eps) * 2)
    return ((-e).exp() * kpt_mask[:, None]).sum(-1) / (kpt_mask.sum(-1)[:, None] + eps)


def _get_covariance_matrix(
    boxes: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    gbbs = torch.cat((boxes[:, 2:4].pow(2) / 12, boxes[:, 4:]), dim=-1)
    a, b, c = gbbs.split(1, dim=-1)
    cos = c.cos()
    sin = c.sin()
    cos2 = cos.pow(2)
    sin2 = sin.pow(2)
    return (a * cos2 + b * sin2, a * sin2 + b * cos2, (a - b) * cos * sin)


def probiou(
    obb1: torch.Tensor, obb2: torch.Tensor, CIoU: bool = False, eps: float = 1e-07
) -> torch.Tensor:
    x1, y1 = obb1[..., :2].split(1, dim=-1)
    x2, y2 = obb2[..., :2].split(1, dim=-1)
    a1, b1, c1 = _get_covariance_matrix(obb1)
    a2, b2, c2 = _get_covariance_matrix(obb2)
    t1 = (
        ((a1 + a2) * (y1 - y2).pow(2) + (b1 + b2) * (x1 - x2).pow(2))
        / ((a1 + a2) * (b1 + b2) - (c1 + c2).pow(2) + eps)
        * 0.25
    )
    t2 = (
        (c1 + c2)
        * (x2 - x1)
        * (y1 - y2)
        / ((a1 + a2) * (b1 + b2) - (c1 + c2).pow(2) + eps)
        * 0.5
    )
    t3 = (
        ((a1 + a2) * (b1 + b2) - (c1 + c2).pow(2))
        / (
            4
            * ((a1 * b1 - c1.pow(2)).clamp_(0) * (a2 * b2 - c2.pow(2)).clamp_(0)).sqrt()
            + eps
        )
        + eps
    ).log() * 0.5
    bd = (t1 + t2 + t3).clamp(eps, 100.0)
    hd = (1.0 - (-bd).exp() + eps).sqrt()
    iou = 1 - hd
    if CIoU:
        w1, h1 = obb1[..., 2:4].split(1, dim=-1)
        w2, h2 = obb2[..., 2:4].split(1, dim=-1)
        v = 4 / math.pi**2 * ((w2 / h2).atan() - (w1 / h1).atan()).pow(2)
        with torch.no_grad():
            alpha = v / (v - iou + (1 + eps))
        return iou - v * alpha
    return iou


def batch_probiou(
    obb1: torch.Tensor | np.ndarray, obb2: torch.Tensor | np.ndarray, eps: float = 1e-07
) -> torch.Tensor:
    obb1 = torch.from_numpy(obb1) if isinstance(obb1, np.ndarray) else obb1
    obb2 = torch.from_numpy(obb2) if isinstance(obb2, np.ndarray) else obb2
    x1, y1 = obb1[..., :2].split(1, dim=-1)
    x2, y2 = (x.squeeze(-1)[None] for x in obb2[..., :2].split(1, dim=-1))
    a1, b1, c1 = _get_covariance_matrix(obb1)
    a2, b2, c2 = (x.squeeze(-1)[None] for x in _get_covariance_matrix(obb2))
    t1 = (
        ((a1 + a2) * (y1 - y2).pow(2) + (b1 + b2) * (x1 - x2).pow(2))
        / ((a1 + a2) * (b1 + b2) - (c1 + c2).pow(2) + eps)
        * 0.25
    )
    t2 = (
        (c1 + c2)
        * (x2 - x1)
        * (y1 - y2)
        / ((a1 + a2) * (b1 + b2) - (c1 + c2).pow(2) + eps)
        * 0.5
    )
    t3 = (
        ((a1 + a2) * (b1 + b2) - (c1 + c2).pow(2))
        / (
            4
            * ((a1 * b1 - c1.pow(2)).clamp_(0) * (a2 * b2 - c2.pow(2)).clamp_(0)).sqrt()
            + eps
        )
        + eps
    ).log() * 0.5
    bd = (t1 + t2 + t3).clamp(eps, 100.0)
    hd = (1.0 - (-bd).exp() + eps).sqrt()
    return 1 - hd


def smooth_bce(eps: float = 0.1) -> tuple[float, float]:
    return (1.0 - 0.5 * eps, 0.5 * eps)


class ConfusionMatrix(DataExportMixin):

    def __init__(
        self,
        names: dict[int, str] = {},
        task: str = "detect",
        save_matches: bool = False,
    ):
        self.task = task
        self.nc = len(names)
        self.matrix = (
            np.zeros((self.nc, self.nc))
            if self.task == "classify"
            else np.zeros((self.nc + 1, self.nc + 1))
        )
        self.names = names
        self.matches = {} if save_matches else None

    def _append_matches(self, mtype: str, batch: dict[str, Any], idx: int) -> None:
        if self.matches is None:
            return
        for k, v in batch.items():
            if k in {"bboxes", "cls", "conf", "keypoints"}:
                self.matches[mtype][k] += v[[idx]]
            elif k == "masks":
                self.matches[mtype][k] += (
                    [v[0] == idx + 1] if v.max() > 1.0 else [v[idx]]
                )

    def process_cls_preds(
        self, preds: list[torch.Tensor], targets: list[torch.Tensor]
    ) -> None:
        preds, targets = (torch.cat(preds)[:, 0], torch.cat(targets))
        for p, t in zip(preds.cpu().numpy(), targets.cpu().numpy()):
            self.matrix[p][t] += 1

    def process_batch(
        self,
        detections: dict[str, torch.Tensor],
        batch: dict[str, Any],
        conf: float = 0.25,
        iou_thres: float = 0.45,
    ) -> None:
        gt_cls, gt_bboxes = (batch["cls"], batch["bboxes"])
        if self.matches is not None:
            self.matches = {k: defaultdict(list) for k in {"TP", "FP", "FN", "GT"}}
            for i in range(gt_cls.shape[0]):
                self._append_matches("GT", batch, i)
        is_obb = gt_bboxes.shape[1] == 5
        conf = 0.25 if conf in {None, 0.01 if is_obb else 0.001} else conf
        no_pred = detections["cls"].shape[0] == 0
        if gt_cls.shape[0] == 0:
            if not no_pred:
                detections = {
                    k: detections[k][detections["conf"] > conf] for k in detections
                }
                detection_classes = detections["cls"].int().tolist()
                for i, dc in enumerate(detection_classes):
                    self.matrix[dc, self.nc] += 1
                    self._append_matches("FP", detections, i)
            return
        if no_pred:
            gt_classes = gt_cls.int().tolist()
            for i, gc in enumerate(gt_classes):
                self.matrix[self.nc, gc] += 1
                self._append_matches("FN", batch, i)
            return
        detections = {k: detections[k][detections["conf"] > conf] for k in detections}
        gt_classes = gt_cls.int().tolist()
        detection_classes = detections["cls"].int().tolist()
        bboxes = detections["bboxes"]
        iou = batch_probiou(gt_bboxes, bboxes) if is_obb else box_iou(gt_bboxes, bboxes)
        x = torch.where(iou > iou_thres)
        if x[0].shape[0]:
            matches = (
                torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1)
                .cpu()
                .numpy()
            )
            if x[0].shape[0] > 1:
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
        else:
            matches = np.zeros((0, 3))
        n = matches.shape[0] > 0
        m0, m1, _ = matches.transpose().astype(int)
        for i, gc in enumerate(gt_classes):
            j = m0 == i
            if n and sum(j) == 1:
                dc = detection_classes[m1[j].item()]
                self.matrix[dc, gc] += 1
                if dc == gc:
                    self._append_matches("TP", detections, m1[j].item())
                else:
                    self._append_matches("FP", detections, m1[j].item())
                    self._append_matches("FN", batch, i)
            else:
                self.matrix[self.nc, gc] += 1
                self._append_matches("FN", batch, i)
        for i, dc in enumerate(detection_classes):
            if not any(m1 == i):
                self.matrix[dc, self.nc] += 1
                self._append_matches("FP", detections, i)

    def matrix(self):
        return self.matrix

    def tp_fp(self) -> tuple[np.ndarray, np.ndarray]:
        tp = self.matrix.diagonal()
        fp = self.matrix.sum(1) - tp
        return (tp, fp) if self.task == "classify" else (tp[:-1], fp[:-1])

    def plot_matches(self, img: torch.Tensor, im_file: str, save_dir: Path) -> None:
        if not self.matches:
            return
        from .ops import xyxy2xywh
        from .plotting import plot_images

        labels = defaultdict(list)
        for i, mtype in enumerate(["GT", "FP", "TP", "FN"]):
            mbatch = self.matches[mtype]
            if "conf" not in mbatch:
                mbatch["conf"] = torch.tensor(
                    [1.0] * len(mbatch["bboxes"]), device=img.device
                )
            mbatch["batch_idx"] = (
                torch.ones(len(mbatch["bboxes"]), device=img.device) * i
            )
            for k in mbatch:
                labels[k] += mbatch[k]
        labels = {
            k: torch.stack(v, 0) if len(v) else torch.empty(0)
            for (k, v) in labels.items()
        }
        if self.task != "obb" and labels["bboxes"].shape[0]:
            labels["bboxes"] = xyxy2xywh(labels["bboxes"])
        (save_dir / "visualizations").mkdir(parents=True, exist_ok=True)
        plot_images(
            labels,
            img.repeat(4, 1, 1, 1),
            paths=[
                "Ground Truth",
                "False Positives",
                "True Positives",
                "False Negatives",
            ],
            fname=save_dir / "visualizations" / Path(im_file).name,
            names=self.names,
            max_subplots=4,
            conf_thres=0.001,
        )

    @TryExcept(msg="ConfusionMatrix plot failure")
    @plt_settings()
    def plot(self, normalize: bool = True, save_dir: str = "", on_plot=None):
        import matplotlib.pyplot as plt

        array = self.matrix / (
            self.matrix.sum(0).reshape(1, -1) + 1e-09 if normalize else 1
        )
        array[array < 0.005] = np.nan
        fig, ax = plt.subplots(1, 1, figsize=(12, 9))
        names, n = (list(self.names.values()), self.nc)
        if self.nc >= 100:
            k = max(2, self.nc // 60)
            keep_idx = slice(None, None, k)
            names = names[keep_idx]
            array = array[keep_idx, :][:, keep_idx]
            n = (self.nc + k - 1) // k
        nc = n if self.task == "classify" else n + 1
        ticklabels = "auto"
        if 0 < nc < 99:
            ticklabels = names if self.task == "classify" else [*names, "background"]
        xy_ticks = np.arange(len(ticklabels)) if ticklabels != "auto" else np.arange(nc)
        tick_fontsize = max(6, 15 - 0.1 * nc)
        label_fontsize = max(6, 12 - 0.1 * nc)
        title_fontsize = max(6, 12 - 0.1 * nc)
        btm = max(0.1, 0.25 - 0.001 * nc)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            im = ax.imshow(array, cmap="Blues", vmin=0.0, interpolation="none")
            ax.xaxis.set_label_position("bottom")
            if nc < 30:
                color_threshold = 0.45 * (1 if normalize else np.nanmax(array))
                for i, row in enumerate(array[:nc]):
                    for j, val in enumerate(row[:nc]):
                        val = array[i, j]
                        if np.isnan(val):
                            continue
                        ax.text(
                            j,
                            i,
                            f"{val:.2f}" if normalize else f"{int(val)}",
                            ha="center",
                            va="center",
                            fontsize=10,
                            color="white" if val > color_threshold else "black",
                        )
            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.05)
        title = "Confusion Matrix" + " Normalized" * normalize
        ax.set_xlabel("True", fontsize=label_fontsize, labelpad=10)
        ax.set_ylabel("Predicted", fontsize=label_fontsize, labelpad=10)
        ax.set_title(title, fontsize=title_fontsize, pad=20)
        ax.set_xticks(xy_ticks)
        ax.set_yticks(xy_ticks)
        ax.tick_params(
            axis="x", bottom=True, top=False, labelbottom=True, labeltop=False
        )
        ax.tick_params(
            axis="y", left=True, right=False, labelleft=True, labelright=False
        )
        if ticklabels != "auto":
            ax.set_xticklabels(
                ticklabels, fontsize=tick_fontsize, rotation=90, ha="center"
            )
            ax.set_yticklabels(ticklabels, fontsize=tick_fontsize)
        for s in {"left", "right", "bottom", "top", "outline"}:
            if s != "outline":
                ax.spines[s].set_visible(False)
            cbar.ax.spines[s].set_visible(False)
        fig.subplots_adjust(left=0, right=0.84, top=0.94, bottom=btm)
        plot_fname = Path(save_dir) / f"{title.lower().replace(' ', '_')}.png"
        fig.savefig(plot_fname, dpi=250)
        plt.close(fig)
        if on_plot:
            on_plot(
                plot_fname, {"type": "confusion_matrix", "matrix": self.matrix.tolist()}
            )

    def print(self):
        for i in range(self.matrix.shape[0]):
            LOGGER.info(" ".join(map(str, self.matrix[i])))

    def summary(
        self, normalize: bool = False, decimals: int = 5
    ) -> list[dict[str, float]]:
        import re

        names = (
            list(self.names.values())
            if self.task == "classify"
            else [*list(self.names.values()), "background"]
        )
        clean_names, seen = ([], set())
        for name in names:
            clean_name = re.sub("[^a-zA-Z0-9_]", "_", name)
            original_clean = clean_name
            counter = 1
            while clean_name.lower() in seen:
                clean_name = f"{original_clean}_{counter}"
                counter += 1
            seen.add(clean_name.lower())
            clean_names.append(clean_name)
        array = (
            self.matrix
            / (self.matrix.sum(0).reshape(1, -1) + 1e-09 if normalize else 1)
        ).round(decimals)
        return [
            dict(
                {"Predicted": clean_names[i]},
                **{clean_names[j]: array[i, j] for j in range(len(clean_names))},
            )
            for i in range(len(clean_names))
        ]


def smooth(y: np.ndarray, f: float = 0.05) -> np.ndarray:
    nf = round(len(y) * f * 2) // 2 + 1
    p = np.ones(nf // 2)
    yp = np.concatenate((p * y[0], y, p * y[-1]), 0)
    return np.convolve(yp, np.ones(nf) / nf, mode="valid")


@plt_settings()
def plot_pr_curve(
    px: np.ndarray,
    py: np.ndarray,
    ap: np.ndarray,
    save_dir: Path = Path("pr_curve.png"),
    names: dict[int, str] = {},
    on_plot=None,
):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True)
    py = np.stack(py, axis=1)
    if 0 < len(names) < 21:
        for i, y in enumerate(py.T):
            ax.plot(px, y, linewidth=1, label=f"{names[i]} {ap[i, 0]:.3f}")
    else:
        ax.plot(px, py, linewidth=1, color="gray")
    ax.plot(
        px,
        py.mean(1),
        linewidth=3,
        color="blue",
        label=f"all classes {ap[:, 0].mean():.3f} mAP@0.5",
    )
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    ax.set_title("Precision-Recall Curve")
    fig.savefig(save_dir, dpi=250)
    plt.close(fig)
    if on_plot:
        on_plot(
            save_dir,
            {
                "type": "pr_curve",
                "x": px.tolist(),
                "y": py.T.tolist(),
                "ap": ap.tolist(),
            },
        )


@plt_settings()
def plot_mc_curve(
    px: np.ndarray,
    py: np.ndarray,
    save_dir: Path = Path("mc_curve.png"),
    names: dict[int, str] = {},
    xlabel: str = "Confidence",
    ylabel: str = "Metric",
    on_plot=None,
):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True)
    if 0 < len(names) < 21:
        for i, y in enumerate(py):
            ax.plot(px, y, linewidth=1, label=f"{names[i]}")
    else:
        ax.plot(px, py.T, linewidth=1, color="gray")
    y = smooth(py.mean(0), 0.1)
    ax.plot(
        px,
        y,
        linewidth=3,
        color="blue",
        label=f"all classes {y.max():.2f} at {px[y.argmax()]:.3f}",
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    ax.set_title(f"{ylabel}-Confidence Curve")
    fig.savefig(save_dir, dpi=250)
    plt.close(fig)
    if on_plot:
        on_plot(
            save_dir,
            {"type": f"{ylabel.lower()}_curve", "x": px.tolist(), "y": py.tolist()},
        )


def compute_ap(
    recall: list[float], precision: list[float]
) -> tuple[float, np.ndarray, np.ndarray]:
    mrec = np.concatenate(([0.0], recall, [recall[-1] if len(recall) else 1.0], [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0], [0.0]))
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
    method = "interp"
    if method == "interp":
        x = np.linspace(0, 1, 101)
        func = (
            np.trapezoid if checks.check_version(np.__version__, ">=2.0") else np.trapz
        )
        ap = func(np.interp(x, mrec, mpre), x)
    else:
        i = np.where(mrec[1:] != mrec[:-1])[0]
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return (ap, mpre, mrec)


def ap_per_class(
    tp: np.ndarray,
    conf: np.ndarray,
    pred_cls: np.ndarray,
    target_cls: np.ndarray,
    plot: bool = False,
    on_plot=None,
    save_dir: Path = Path(),
    names: dict[int, str] = {},
    eps: float = 1e-16,
    prefix: str = "",
) -> tuple:
    i = np.argsort(-conf)
    tp, conf, pred_cls = (tp[i], conf[i], pred_cls[i])
    unique_classes, nt = np.unique(target_cls, return_counts=True)
    nc = unique_classes.shape[0]
    x, prec_values = (np.linspace(0, 1, 1000), [])
    ap, p_curve, r_curve = (
        np.zeros((nc, tp.shape[1])),
        np.zeros((nc, 1000)),
        np.zeros((nc, 1000)),
    )
    for ci, c in enumerate(unique_classes):
        i = pred_cls == c
        n_l = nt[ci]
        n_p = i.sum()
        if n_p == 0 or n_l == 0:
            continue
        fpc = (1 - tp[i]).cumsum(0)
        tpc = tp[i].cumsum(0)
        recall = tpc / (n_l + eps)
        r_curve[ci] = np.interp(-x, -conf[i], recall[:, 0], left=0)
        precision = tpc / (tpc + fpc)
        p_curve[ci] = np.interp(-x, -conf[i], precision[:, 0], left=1)
        for j in range(tp.shape[1]):
            ap[ci, j], mpre, mrec = compute_ap(recall[:, j], precision[:, j])
            if j == 0:
                prec_values.append(np.interp(x, mrec, mpre))
    prec_values = np.array(prec_values) if prec_values else np.zeros((1, 1000))
    f1_curve = 2 * p_curve * r_curve / (p_curve + r_curve + eps)
    names = {i: names[k] for (i, k) in enumerate(unique_classes) if k in names}
    if plot:
        plot_pr_curve(
            x,
            prec_values,
            ap,
            save_dir / f"{prefix}PR_curve.png",
            names,
            on_plot=on_plot,
        )
        plot_mc_curve(
            x,
            f1_curve,
            save_dir / f"{prefix}F1_curve.png",
            names,
            ylabel="F1",
            on_plot=on_plot,
        )
        plot_mc_curve(
            x,
            p_curve,
            save_dir / f"{prefix}P_curve.png",
            names,
            ylabel="Precision",
            on_plot=on_plot,
        )
        plot_mc_curve(
            x,
            r_curve,
            save_dir / f"{prefix}R_curve.png",
            names,
            ylabel="Recall",
            on_plot=on_plot,
        )
    i = smooth(f1_curve.mean(0), 0.1).argmax()
    p, r, f1 = (p_curve[:, i], r_curve[:, i], f1_curve[:, i])
    tp = (r * nt).round()
    fp = (tp / (p + eps) - tp).round()
    return (
        tp,
        fp,
        p,
        r,
        f1,
        ap,
        unique_classes.astype(int),
        p_curve,
        r_curve,
        f1_curve,
        x,
        prec_values,
    )


class Metric(SimpleClass):

    def __init__(self) -> None:
        self.p = []
        self.r = []
        self.f1 = []
        self.all_ap = []
        self.ap_class_index = []
        self.nc = 0
        self.image_metrics = {}

    @property
    def ap50(self) -> np.ndarray | list:
        return self.all_ap[:, 0] if len(self.all_ap) else []

    @property
    def ap(self) -> np.ndarray | list:
        return self.all_ap.mean(1) if len(self.all_ap) else []

    @property
    def mp(self) -> float:
        return self.p.mean() if len(self.p) else 0.0

    @property
    def mr(self) -> float:
        return self.r.mean() if len(self.r) else 0.0

    @property
    def map50(self) -> float:
        return self.all_ap[:, 0].mean() if len(self.all_ap) else 0.0

    @property
    def map75(self) -> float:
        return self.all_ap[:, 5].mean() if len(self.all_ap) else 0.0

    @property
    def map(self) -> float:
        return self.all_ap.mean() if len(self.all_ap) else 0.0

    def mean_results(self) -> list[float]:
        return [self.mp, self.mr, self.map50, self.map]

    def class_result(self, i: int) -> tuple[float, float, float, float]:
        return (self.p[i], self.r[i], self.ap50[i], self.ap[i])

    @property
    def maps(self) -> np.ndarray:
        maps = np.zeros(self.nc) + self.map
        for i, c in enumerate(self.ap_class_index):
            maps[c] = self.ap[i]
        return maps

    def fitness(self) -> float:
        w = [0.0, 0.0, 0.0, 1.0]
        return float((np.nan_to_num(np.array(self.mean_results())) * w).sum())

    def update(self, results: tuple):
        (
            self.p,
            self.r,
            self.f1,
            self.all_ap,
            self.ap_class_index,
            self.p_curve,
            self.r_curve,
            self.f1_curve,
            self.px,
            self.prec_values,
        ) = results

    def clear_image_metrics(self) -> None:
        self.image_metrics.clear()

    @property
    def curves(self) -> list:
        return []

    @property
    def curves_results(self) -> list[list]:
        return [
            [self.px, self.prec_values, "Recall", "Precision"],
            [self.px, self.f1_curve, "Confidence", "F1"],
            [self.px, self.p_curve, "Confidence", "Precision"],
            [self.px, self.r_curve, "Confidence", "Recall"],
        ]

    def update_image_metrics(
        self, tp: np.ndarray, target_cls: np.ndarray, pred_cls: np.ndarray, im_name: str
    ) -> None:
        tp = int(tp[:, 0].sum())
        num_preds = pred_cls.shape[0]
        num_targets = target_cls.shape[0]
        fp = num_preds - tp
        fn = num_targets - tp
        if num_preds == 0 and num_targets == 0:
            precision = recall = f1 = 1.0
        else:
            precision = tp / num_preds if num_preds else 0.0
            recall = tp / num_targets if num_targets else 0.0
            denom = precision + recall
            f1 = 2 * precision * recall / denom if denom else 0.0
        self.image_metrics[im_name] = {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
        }


class DetMetrics(SimpleClass, DataExportMixin):

    def __init__(self, names: dict[int, str] = {}) -> None:
        self.names = names
        self.box = Metric()
        self.speed = {
            "preprocess": 0.0,
            "inference": 0.0,
            "loss": 0.0,
            "postprocess": 0.0,
        }
        self.stats = dict(tp=[], conf=[], pred_cls=[], target_cls=[], target_img=[])
        self.nt_per_class = None
        self.nt_per_image = None

    def update_stats(self, stat: dict[str, Any]) -> None:
        for k in self.stats.keys():
            self.stats[k].append(stat[k])
        self.box.update_image_metrics(
            stat["tp"], stat["target_cls"], stat["pred_cls"], stat["im_name"]
        )

    def process(
        self, save_dir: Path = Path("."), plot: bool = False, on_plot=None
    ) -> dict[str, np.ndarray]:
        stats = {k: np.concatenate(v, 0) for (k, v) in self.stats.items()}
        if not stats:
            return stats
        results = ap_per_class(
            stats["tp"],
            stats["conf"],
            stats["pred_cls"],
            stats["target_cls"],
            plot=plot,
            save_dir=save_dir,
            names=self.names,
            on_plot=on_plot,
            prefix="Box",
        )[2:]
        self.box.nc = len(self.names)
        self.box.update(results)
        self.nt_per_class = np.bincount(
            stats["target_cls"].astype(int), minlength=len(self.names)
        )
        self.nt_per_image = np.bincount(
            stats["target_img"].astype(int), minlength=len(self.names)
        )
        return stats

    def clear_stats(self):
        for v in self.stats.values():
            v.clear()

    def clear_image_metrics(self) -> None:
        self.box.clear_image_metrics()

    @property
    def keys(self) -> list[str]:
        return [
            "metrics/precision(B)",
            "metrics/recall(B)",
            "metrics/mAP50(B)",
            "metrics/mAP50-95(B)",
        ]

    def mean_results(self) -> list[float]:
        return self.box.mean_results()

    def class_result(self, i: int) -> tuple[float, float, float, float]:
        return self.box.class_result(i)

    @property
    def maps(self) -> np.ndarray:
        return self.box.maps

    @property
    def fitness(self) -> float:
        return self.box.fitness()

    @property
    def ap_class_index(self) -> list:
        return self.box.ap_class_index

    @property
    def results_dict(self) -> dict[str, float]:
        keys = [*self.keys, "fitness"]
        values = (
            float(x) if hasattr(x, "item") else x
            for x in [*self.mean_results(), self.fitness]
        )
        return dict(zip(keys, values))

    @property
    def curves(self) -> list[str]:
        return [
            "Precision-Recall(B)",
            "F1-Confidence(B)",
            "Precision-Confidence(B)",
            "Recall-Confidence(B)",
        ]

    @property
    def curves_results(self) -> list[list]:
        return self.box.curves_results

    def summary(
        self, normalize: bool = True, decimals: int = 5
    ) -> list[dict[str, Any]]:
        per_class = {"Box-P": self.box.p, "Box-R": self.box.r, "Box-F1": self.box.f1}
        return [
            {
                "Class": self.names[self.ap_class_index[i]],
                "Images": self.nt_per_image[self.ap_class_index[i]],
                "Instances": self.nt_per_class[self.ap_class_index[i]],
                **{k: round(v[i], decimals) for (k, v) in per_class.items()},
                "mAP50": round(self.class_result(i)[2], decimals),
                "mAP50-95": round(self.class_result(i)[3], decimals),
            }
            for i in range(len(per_class["Box-P"]))
        ]


class SegmentMetrics(DetMetrics):

    def __init__(self, names: dict[int, str] = {}) -> None:
        DetMetrics.__init__(self, names)
        self.seg = Metric()
        self.stats["tp_m"] = []

    def update_stats(self, stat: dict[str, Any]) -> None:
        super().update_stats(stat)
        self.seg.update_image_metrics(
            stat["tp_m"], stat["target_cls"], stat["pred_cls"], stat["im_name"]
        )

    def clear_image_metrics(self) -> None:
        super().clear_image_metrics()
        self.seg.clear_image_metrics()

    def process(
        self, save_dir: Path = Path("."), plot: bool = False, on_plot=None
    ) -> dict[str, np.ndarray]:
        stats = DetMetrics.process(self, save_dir, plot, on_plot=on_plot)
        results_mask = ap_per_class(
            stats["tp_m"],
            stats["conf"],
            stats["pred_cls"],
            stats["target_cls"],
            plot=plot,
            on_plot=on_plot,
            save_dir=save_dir,
            names=self.names,
            prefix="Mask",
        )[2:]
        self.seg.nc = len(self.names)
        self.seg.update(results_mask)
        return stats

    @property
    def keys(self) -> list[str]:
        return [
            *DetMetrics.keys.fget(self),
            "metrics/precision(M)",
            "metrics/recall(M)",
            "metrics/mAP50(M)",
            "metrics/mAP50-95(M)",
        ]

    def mean_results(self) -> list[float]:
        return DetMetrics.mean_results(self) + self.seg.mean_results()

    def class_result(self, i: int) -> list[float]:
        return DetMetrics.class_result(self, i) + self.seg.class_result(i)

    @property
    def maps(self) -> np.ndarray:
        return DetMetrics.maps.fget(self) + self.seg.maps

    @property
    def fitness(self) -> float:
        return self.seg.fitness() + DetMetrics.fitness.fget(self)

    @property
    def curves(self) -> list[str]:
        return [
            *DetMetrics.curves.fget(self),
            "Precision-Recall(M)",
            "F1-Confidence(M)",
            "Precision-Confidence(M)",
            "Recall-Confidence(M)",
        ]

    @property
    def curves_results(self) -> list[list]:
        return DetMetrics.curves_results.fget(self) + self.seg.curves_results

    def summary(
        self, normalize: bool = True, decimals: int = 5
    ) -> list[dict[str, Any]]:
        per_class = {"Mask-P": self.seg.p, "Mask-R": self.seg.r, "Mask-F1": self.seg.f1}
        summary = DetMetrics.summary(self, normalize, decimals)
        for i, s in enumerate(summary):
            s.update({**{k: round(v[i], decimals) for (k, v) in per_class.items()}})
        return summary


class PoseMetrics(DetMetrics):

    def __init__(self, names: dict[int, str] = {}) -> None:
        super().__init__(names)
        self.pose = Metric()
        self.stats["tp_p"] = []

    def update_stats(self, stat: dict[str, Any]) -> None:
        super().update_stats(stat)
        self.pose.update_image_metrics(
            stat["tp_p"], stat["target_cls"], stat["pred_cls"], stat["im_name"]
        )

    def clear_image_metrics(self) -> None:
        super().clear_image_metrics()
        self.pose.clear_image_metrics()

    def process(
        self, save_dir: Path = Path("."), plot: bool = False, on_plot=None
    ) -> dict[str, np.ndarray]:
        stats = DetMetrics.process(self, save_dir, plot, on_plot=on_plot)
        results_pose = ap_per_class(
            stats["tp_p"],
            stats["conf"],
            stats["pred_cls"],
            stats["target_cls"],
            plot=plot,
            on_plot=on_plot,
            save_dir=save_dir,
            names=self.names,
            prefix="Pose",
        )[2:]
        self.pose.nc = len(self.names)
        self.pose.update(results_pose)
        return stats

    @property
    def keys(self) -> list[str]:
        return [
            *DetMetrics.keys.fget(self),
            "metrics/precision(P)",
            "metrics/recall(P)",
            "metrics/mAP50(P)",
            "metrics/mAP50-95(P)",
        ]

    def mean_results(self) -> list[float]:
        return DetMetrics.mean_results(self) + self.pose.mean_results()

    def class_result(self, i: int) -> list[float]:
        return DetMetrics.class_result(self, i) + self.pose.class_result(i)

    @property
    def maps(self) -> np.ndarray:
        return DetMetrics.maps.fget(self) + self.pose.maps

    @property
    def fitness(self) -> float:
        return self.pose.fitness() + DetMetrics.fitness.fget(self)

    @property
    def curves(self) -> list[str]:
        return [
            *DetMetrics.curves.fget(self),
            "Precision-Recall(B)",
            "F1-Confidence(B)",
            "Precision-Confidence(B)",
            "Recall-Confidence(B)",
            "Precision-Recall(P)",
            "F1-Confidence(P)",
            "Precision-Confidence(P)",
            "Recall-Confidence(P)",
        ]

    @property
    def curves_results(self) -> list[list]:
        return DetMetrics.curves_results.fget(self) + self.pose.curves_results

    def summary(
        self, normalize: bool = True, decimals: int = 5
    ) -> list[dict[str, Any]]:
        per_class = {
            "Pose-P": self.pose.p,
            "Pose-R": self.pose.r,
            "Pose-F1": self.pose.f1,
        }
        summary = DetMetrics.summary(self, normalize, decimals)
        for i, s in enumerate(summary):
            s.update({**{k: round(v[i], decimals) for (k, v) in per_class.items()}})
        return summary


class ClassifyMetrics(SimpleClass, DataExportMixin):

    def __init__(self) -> None:
        self.top1 = 0
        self.top5 = 0
        self.speed = {
            "preprocess": 0.0,
            "inference": 0.0,
            "loss": 0.0,
            "postprocess": 0.0,
        }

    def process(self, targets: torch.Tensor, pred: torch.Tensor):
        pred, targets = (torch.cat(pred), torch.cat(targets))
        correct = (targets[:, None] == pred).float()
        acc = torch.stack((correct[:, 0], correct.max(1).values), dim=1)
        self.top1, self.top5 = acc.mean(0).tolist()

    @property
    def fitness(self) -> float:
        return (self.top1 + self.top5) / 2

    @property
    def results_dict(self) -> dict[str, float]:
        return dict(zip([*self.keys, "fitness"], [self.top1, self.top5, self.fitness]))

    @property
    def keys(self) -> list[str]:
        return ["metrics/accuracy_top1", "metrics/accuracy_top5"]

    @property
    def curves(self) -> list:
        return []

    @property
    def curves_results(self) -> list:
        return []

    def summary(
        self, normalize: bool = True, decimals: int = 5
    ) -> list[dict[str, float]]:
        return [
            {
                "top1_acc": round(self.top1, decimals),
                "top5_acc": round(self.top5, decimals),
            }
        ]


class OBBMetrics(DetMetrics):

    def __init__(self, names: dict[int, str] = {}) -> None:
        DetMetrics.__init__(self, names)
