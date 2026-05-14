from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.utils.data as data
import yaml
from tqdm import tqdm

from data.config import cfg
from data.source_domain import SourceDomainDetection, detection_collate
from models.factory import build_net
from utils import visualize as viz

from train import (
    _BACKBONE_FROM_MODEL,
    _MODEL_FROM_ARCH_BACKBONE,
    _decode_per_image,
    _detect_metrics_from_cm,
    _format_val_table,
    build_dark_batch,
    collect_target_samples,
    infer_detections,
    viz_config,
    viz_method,
)


_TEST_DEFAULTS: Dict[str, Any] = {
    'batch_size': 1,
    'num_workers': 0,
    'cuda': True,
    'val_file': './dataset/source_val.txt',
    'nc': 3,
    'source_folder': '',
    'target_folder': '/media/caotulab/303A225B3A221DFA/Nhan/data/images/target',
    'charts_dir': './charts',
    'mode_name': 'test',
    'viz_num_samples': 6,
    'weights': None,
    'iou_thr': 0.5,
    'score_thr_cm': 0.5,
}


def load_test_config(config_path: str) -> argparse.Namespace:
    """Load configs/test/<arch>/<backbone>/<num_exp>.yaml into a Namespace."""
    p = Path(config_path)
    if not p.is_file():
        raise FileNotFoundError(f'Config not found: {config_path}')
    with p.open() as f:
        cfg_yaml = yaml.safe_load(f) or {}
    if not isinstance(cfg_yaml, dict):
        raise ValueError(f'Top-level YAML must be a mapping, got {type(cfg_yaml)}')

    parts = p.parts
    path_arch = parts[-3] if len(parts) >= 3 else None
    path_backbone = parts[-2] if len(parts) >= 2 else None
    path_num_exp = p.stem

    arch = cfg_yaml.get('architecture') or path_arch
    backbone = cfg_yaml.get('backbone') or path_backbone
    num_exp = cfg_yaml.get('num_exp') or path_num_exp

    model = cfg_yaml.get('model') or _MODEL_FROM_ARCH_BACKBONE.get((arch, backbone))
    if model is None:
        raise ValueError(
            f'Cannot infer --model from architecture={arch!r}, '
            f'backbone={backbone!r}. Add `model:` to {config_path}.'
        )

    merged: Dict[str, Any] = dict(_TEST_DEFAULTS)
    for k, v in cfg_yaml.items():
        if k in ('architecture', 'backbone', 'num_exp', 'model'):
            continue
        merged[k] = v
    merged.update(dict(
        architecture=arch, backbone=backbone, model=model, num_exp=num_exp,
        config=str(p),
    ))

    if not merged.get('weights'):
        raise ValueError(
            f'`weights:` must be set in {config_path} (path to a trained '
            f'.pth checkpoint).'
        )
    if not os.path.isfile(merged['weights']):
        raise FileNotFoundError(
            f'weights file does not exist: {merged["weights"]}'
        )
    return argparse.Namespace(**merged)


def _load_state_dict(net: torch.nn.Module, weights_path: str) -> None:
    state = torch.load(weights_path, map_location='cpu')
    if isinstance(state, dict) and 'weight' in state:
        # Came from last_model.pth which wraps {'epoch', 'weight'}.
        state = state['weight']
    net.load_state_dict(state)


def evaluate(args_ns: argparse.Namespace) -> Dict[str, float]:
    use_cuda = bool(args_ns.cuda) and torch.cuda.is_available()
    if use_cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')

    num_classes = args_ns.nc + 1  # +1 for background
    cfg.NUM_CLASSES = num_classes

    print(f'[net] building {args_ns.architecture}/{args_ns.model}')
    net = build_net('train', num_classes, args_ns.model)
    _load_state_dict(net, args_ns.weights)
    print(f'[net] loaded weights from {args_ns.weights}')
    if use_cuda:
        net = net.cuda()
    net.eval()

    val_ds = SourceDomainDetection(args_ns.val_file, mode='val')
    val_loader = data.DataLoader(
        val_ds, batch_size=args_ns.batch_size,
        num_workers=args_ns.num_workers,
        collate_fn=detection_collate,
        shuffle=False, pin_memory=use_cuda,
    )
    print(f'[data] val samples: {len(val_ds)} | batch_size: {args_ns.batch_size}')

    per_image: List[Dict[str, np.ndarray]] = []
    pbar = tqdm(val_loader, total=len(val_loader),
                desc='Evaluating', dynamic_ncols=True,
                unit='batch', colour='green')

    t0 = time.time()
    with torch.no_grad():
        for images, targets, _ in pbar:
            if use_cuda:
                images = images.cuda()
            images = images / 255.0
            img_dark = build_dark_batch(images)
            out, _ = net.test_forward(img_dark)
            per_image.extend(_decode_per_image(out, targets, net))
    elapsed = time.time() - t0

    scores, matched, n_gt, cm = viz.evaluate_detections(
        per_image,
        iou_thr=float(args_ns.iou_thr),
        score_thr_cm=float(args_ns.score_thr_cm),
    )
    m = _detect_metrics_from_cm(cm)
    _, _, _, mAP = viz._pr_from_scores(
        np.asarray(scores), np.asarray(matched), n_gt,
    )

    table = _format_val_table('test', [
        ('weights',   os.path.basename(args_ns.weights)),
        ('n_samples', f'{len(val_ds)}'),
        ('n_gt',      f'{n_gt}'),
        ('accuracy',  f'{m["accuracy"]:.4f}'),
        ('precision', f'{m["precision"]:.4f}'),
        ('recall',    f'{m["recall"]:.4f}'),
        ('f1',        f'{m["f1"]:.4f}'),
        ('mAP@0.5',   f'{mAP:.4f}'),
        ('TP/FP/FN',  f'{m["tp"]}/{m["fp"]}/{m["fn"]}'),
        ('iou_thr',   f'{float(args_ns.iou_thr):.2f}'),
        ('score_thr', f'{float(args_ns.score_thr_cm):.2f}'),
        ('time(s)',   f'{elapsed:.2f}'),
    ])
    print(table)

    # Charts dir: charts/<mode_name>/<arch>/<backbone>/<num_exp>/
    backbone_dir = _BACKBONE_FROM_MODEL.get(args_ns.model, args_ns.model)
    charts_dir = viz.make_charts_dir(
        args_ns.charts_dir, args_ns.mode_name,
        args_ns.architecture, backbone_dir, args_ns.num_exp,
    )

    method = viz_method()
    config_dict = viz_config(args_ns, {
        'weights': os.path.basename(args_ns.weights),
        'iou_thr': float(args_ns.iou_thr),
    })

    viz.plot_confusion_matrix(cm, charts_dir,
                              method=method, config=config_dict,
                              classes=('background', 'object'))
    viz.plot_pr_curve(scores, matched, n_gt, charts_dir,
                     method=method, config=config_dict)
    viz.plot_recall_f1_curve(scores, matched, n_gt, charts_dir,
                             method=method, config=config_dict)

    # Sample-prediction figures (day / synth-night / real-night).
    day_samples, synth_night_samples = _collect_paired_samples(
        net, val_loader, args_ns.viz_num_samples, use_cuda,
    )
    if day_samples:
        viz.plot_sample_predictions(
            day_samples, charts_dir, 'samples_day.png',
            method=method, config=config_dict,
            title_suffix='(val / day)',
        )
    if synth_night_samples:
        viz.plot_sample_predictions(
            synth_night_samples, charts_dir, 'samples_synth_night.png',
            method=method, config=config_dict,
            title_suffix='(val + Dark ISP)',
        )

    real_night_samples = collect_target_samples(
        net, args_ns.target_folder, args_ns.viz_num_samples,
    )
    if real_night_samples:
        viz.plot_sample_predictions(
            real_night_samples, charts_dir, 'samples_real_night.png',
            method=method, config=config_dict,
            title_suffix='(real target / night)',
        )

    metrics = dict(
        accuracy=m['accuracy'], precision=m['precision'],
        recall=m['recall'], f1=m['f1'], mAP=float(mAP),
        tp=m['tp'], fp=m['fp'], fn=m['fn'], n_gt=int(n_gt),
        iou_thr=float(args_ns.iou_thr),
        score_thr=float(args_ns.score_thr_cm),
        weights=str(args_ns.weights),
        elapsed_s=elapsed,
    )
    with open(os.path.join(charts_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f'[viz] charts + metrics saved to {charts_dir}')
    return metrics


def _collect_paired_samples(
    net: torch.nn.Module,
    val_loader: data.DataLoader,
    n_show: int,
    use_cuda: bool,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Run a few val batches in day + synth-night form for sample viz."""
    day: List[Dict[str, Any]] = []
    synth: List[Dict[str, Any]] = []
    if n_show <= 0:
        return day, synth
    with torch.no_grad():
        for images, targets, img_paths in val_loader:
            if use_cuda:
                images = images.cuda()
            images = images / 255.0
            img_dark = build_dark_batch(images)
            for i in range(images.shape[0]):
                if len(day) < n_show:
                    pb_d, ps_d = infer_detections(net, images[i])
                    day.append(dict(
                        image=(images[i].detach().cpu().numpy()
                               .transpose(1, 2, 0) * 255
                               ).clip(0, 255).astype(np.uint8),
                        boxes=pb_d, scores=ps_d,
                        labels=['cls'] * len(pb_d),
                        title=os.path.basename(img_paths[i])
                              if i < len(img_paths) else '',
                    ))
                if len(synth) < n_show:
                    pb_n, ps_n = infer_detections(net, img_dark[i])
                    synth.append(dict(
                        image=(img_dark[i].detach().cpu().numpy()
                               .transpose(1, 2, 0) * 255
                               ).clip(0, 255).astype(np.uint8),
                        boxes=pb_n, scores=ps_n,
                        labels=['cls'] * len(pb_n),
                        title='synth/' + (
                            os.path.basename(img_paths[i])
                            if i < len(img_paths) else ''
                        ),
                    ))
            if len(day) >= n_show and len(synth) >= n_show:
                break
    return day, synth


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser('DAI-Net evaluation driven by a YAML config.')
    p.add_argument(
        '--config', required=True, type=str,
        help='Path to YAML config, e.g. '
             'configs/test/dai_net/vgg16/exp1.yaml',
    )
    cli = p.parse_args()
    return load_test_config(cli.config)


def main() -> None:
    args_ns = parse_args()
    evaluate(args_ns)


if __name__ == '__main__':
    main()
