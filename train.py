from __future__ import annotations

import argparse
import csv
import datetime as _dt
import glob
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import yaml

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
from PIL import Image
from torch.autograd import Variable
from torchmetrics.functional.image import structural_similarity_index_measure as ssim
from tqdm import tqdm

from data.config import cfg
from data.source_domain import SourceDomainDetection, detection_collate
from layers.modules import EnhanceLoss, MultiBoxLoss
from models.enhancer import RetinexNet
from models.factory import basenet_factory, build_net
from utils import visualize as viz
from utils.dark_isp import Low_Illumination_Degrading


Point = Tuple[float, float]
History = Dict[str, List[Point]]


CHECKPOINT_LATEST = 'last_model.pth'
CHECKPOINT_BEST = 'best_model.pth'
RETINEX_WEIGHTS = 'decomp.pth'
PRINT_EVERY = 100


_BACKBONE_FROM_MODEL: Dict[str, str] = {
    'dark': 'vgg16',
    'vgg': 'vgg16',
    'resnet50': 'resnet50',
    'resnet101': 'resnet101',
    'resnet152': 'resnet152',
}
_DEFAULT_ARCH_FROM_MODEL: Dict[str, str] = {
    'dark': 'dai_net',
    'vgg': 'dsfd',
    'resnet50': 'dsfd',
    'resnet101': 'dsfd',
    'resnet152': 'dsfd',
}


_MODEL_FROM_ARCH_BACKBONE: Dict[Tuple[str, str], str] = {
    ('dai_net', 'vgg16'): 'dark',
    ('dsfd', 'vgg16'): 'vgg',
    ('dsfd', 'resnet50'): 'resnet50',
    ('dsfd', 'resnet101'): 'resnet101',
    ('dsfd', 'resnet152'): 'resnet152',
}


def resolve_arch_and_backbone(args_ns: argparse.Namespace) -> Tuple[str, str]:
    arch = args_ns.architecture or _DEFAULT_ARCH_FROM_MODEL.get(
        args_ns.model, 'dai_net',
    )
    backbone = _BACKBONE_FROM_MODEL.get(args_ns.model, args_ns.model)
    return arch, backbone


# Defaults for every field the rest of train.py reads off args_ns. YAML values
# override these.
_TRAIN_DEFAULTS: Dict[str, Any] = {
    'batch_size': 4,
    'num_workers': 0,
    'cuda': True,
    'lr': 5e-4,
    'momentum': 0.9,
    'weight_decay': 5e-4,
    'gamma': 0.1,
    'multigpu': True,
    'save_folder': 'weights/',
    'train_file': './dataset/source_train.txt',
    'val_file': './dataset/source_val.txt',
    'nc': 3,
    'source_folder': '/media/caotulab/303A225B3A221DFA/Nhan/data/images/source',
    'target_folder': '/media/caotulab/303A225B3A221DFA/Nhan/data/images/target',
    'charts_dir': './charts',
    'records_dir': './records',
    'viz_num_samples': 6,
    'viz_every_iters': 500,
    'viz_full_every_epochs': 1,
    'resume': None,
    # Training schedule (override the defaults baked into data/config.py).
    'epochs':    100,
    'max_steps': 150000,
    'lr_steps':  [20000, 25000, 30000],
}


_TRAIN_COLUMNS: Tuple[str, ...] = (
    'epoch', 'iteration', 'lr',
    'loss', 'pal1_loc', 'pal1_conf', 'pal2_loc', 'pal2_conf',
    'enhance', 'enhance_l1ssim', 'mutual',
    'elapsed_s', 'timestamp',
)
_VAL_COLUMNS: Tuple[str, ...] = (
    'epoch', 'loss', 'pal2_loc', 'pal2_conf',
    'accuracy', 'precision', 'recall', 'f1', 'mAP',
    'tp', 'fp', 'fn', 'elapsed_s', 'timestamp',
)


def _records_paths(records_root: str, architecture: str,
                   backbone: str, num_exp: str) -> Tuple[str, str]:
    parent = Path(records_root) / architecture / backbone
    parent.mkdir(parents=True, exist_ok=True)
    return (
        str(parent / f'{num_exp}_train.csv'),
        str(parent / f'{num_exp}_val.csv'),
    )


def _append_record_row(path: str, columns: Tuple[str, ...],
                       row: Dict[str, Any]) -> None:
    """Append one row to a CSV — write header on first row of a fresh file."""
    is_new = not os.path.exists(path)
    with open(path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        if is_new:
            writer.writeheader()
        writer.writerow({k: row.get(k, '') for k in columns})


def _resolve_resume(value: Any, save_folder: str, arch: str,
                    backbone: str, *, verbose: bool = True) -> Optional[str]:
    
    if value is None:
        return None
    if isinstance(value, bool):
        truthy = value
        sval: Optional[str] = None
    else:
        sval = str(value).strip()
        low = sval.lower()
        if low in ('', 'false', 'no', '0', 'null', 'none'):
            return None
        truthy = low in ('true', 'auto', 'yes', '1')
    if truthy:
        auto = os.path.join(save_folder, arch, backbone, CHECKPOINT_LATEST)
        if os.path.isfile(auto):
            return auto
        if verbose:
            print(f'[resume] {auto} not found — starting fresh.')
        return None
    return sval


def load_yaml_config(config_path: str, *,
                     mode: str = 'train') -> argparse.Namespace:

    p = Path(config_path)
    if not p.is_file():
        raise FileNotFoundError(f'Config not found: {config_path}')
    with p.open() as f:
        cfg = yaml.safe_load(f) or {}
    if not isinstance(cfg, dict):
        raise ValueError(f'Top-level YAML must be a mapping, got {type(cfg)}')

    parts = p.parts
    path_arch = parts[-3] if len(parts) >= 3 else None
    path_backbone = parts[-2] if len(parts) >= 2 else None
    path_num_exp = p.stem

    arch = cfg.get('architecture') or path_arch
    backbone = cfg.get('backbone') or path_backbone
    num_exp = cfg.get('num_exp') or path_num_exp

    model = cfg.get('model') or _MODEL_FROM_ARCH_BACKBONE.get((arch, backbone))
    if model is None:
        raise ValueError(
            f'Cannot infer --model from architecture={arch!r}, '
            f'backbone={backbone!r}. Add an explicit `model:` to {config_path} '
            f'(one of dark/vgg/resnet50/resnet101/resnet152).'
        )

    merged: Dict[str, Any] = dict(_TRAIN_DEFAULTS)
    for k, v in cfg.items():
        if k in ('architecture', 'backbone', 'num_exp', 'model'):
            continue
        merged[k] = v
    merged.update(dict(
        architecture=arch, model=model, num_exp=num_exp,
        local_rank=int(os.environ.get('LOCAL_RANK', '0')),
        config=str(p),
    ))

    merged['resume'] = _resolve_resume(
        merged.get('resume'), merged['save_folder'], arch, backbone,
        verbose=(merged['local_rank'] == 0 and mode == 'train'),
    )
    return argparse.Namespace(**merged)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser('DAI-Net training driven by a YAML config.')
    p.add_argument(
        '--config', required=True, type=str,
        help='Path to YAML config, e.g. '
             'configs/train/dai_net/vgg16/exp1.yaml',
    )
    cli = p.parse_args()
    return load_yaml_config(cli.config, mode='train')


class Tee:

    def __init__(self, stream: Any, file_handle: Any) -> None:
        self._stream = stream
        self._file = file_handle

    @staticmethod
    def _strip_cr_updates(data: str) -> str:
        if '\r' not in data:
            return data
        out_chunks = []
        for chunk in data.split('\n'):
            if '\r' in chunk:
                chunk = chunk.split('\r', 1)[0]
            out_chunks.append(chunk)
        return '\n'.join(out_chunks)

    def write(self, data: str) -> None:
        self._stream.write(data)
        try:
            clean = self._strip_cr_updates(data)
            if clean:
                self._file.write(clean)
                self._file.flush()
        except Exception:
            pass

    def flush(self) -> None:
        self._stream.flush()
        try:
            self._file.flush()
        except Exception:
            pass

    def __getattr__(self, name: str):
        return getattr(self._stream, name)


def setup_logging(model: str, num_exp: str,
                  args_ns: argparse.Namespace) -> Optional[str]:
    os.makedirs('logs', exist_ok=True)
    ts = _dt.datetime.now().strftime('%Y%m%d_%H%M%S')
    path = os.path.join('logs', f'{ts}_{model}_{num_exp}.log')
    fh = open(path, 'a', buffering=1)
    fh.write(f'# DAI-Net training log — {_dt.datetime.now().isoformat()}\n')
    fh.write(f'# args: {vars(args_ns)}\n')
    sys.stdout = Tee(sys.stdout, fh)
    sys.stderr = Tee(sys.stderr, fh)
    print(f'[log] writing to {path}')
    return path


def setup_distributed(local_rank: int, use_cuda: bool) -> None:
    if not (torch.cuda.is_available() and use_cuda):
        torch.set_default_tensor_type('torch.FloatTensor')
        return
    gpu_num = torch.cuda.device_count()
    if local_rank == 0:
        print(f'Using {gpu_num} gpus')
    rank = int(os.environ.get('RANK', '0'))
    torch.cuda.set_device(rank % gpu_num)
    dist.init_process_group('nccl')


def build_data_loaders(
    args_ns: argparse.Namespace,
) -> Tuple[SourceDomainDetection, data.DataLoader,
           SourceDomainDetection, data.DataLoader]:
    train_ds = SourceDomainDetection(args_ns.train_file, mode='train')
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_ds, shuffle=True,
    )
    train_loader = data.DataLoader(
        train_ds, args_ns.batch_size,
        num_workers=args_ns.num_workers,
        collate_fn=detection_collate,
        sampler=train_sampler,
        pin_memory=True,
    )

    val_ds = SourceDomainDetection(args_ns.val_file, mode='val')
    val_sampler = torch.utils.data.distributed.DistributedSampler(
        val_ds, shuffle=False,
    )
    val_loader = data.DataLoader(
        val_ds, args_ns.batch_size,
        num_workers=0,
        collate_fn=detection_collate,
        sampler=val_sampler,
        pin_memory=True,
    )
    return train_ds, train_loader, val_ds, val_loader


def adjust_learning_rate(optimizer: optim.Optimizer, gamma: float) -> None:
    for g in optimizer.param_groups:
        g['lr'] = g['lr'] * gamma


def build_dark_batch(images: torch.Tensor) -> torch.Tensor:
    """Synthesize paired low-light batch via Dark ISP (paper-faithful)."""
    img_dark = torch.empty_like(images)
    for i in range(images.shape[0]):
        img_dark[i], _ = Low_Illumination_Degrading(images[i])
    return img_dark


def load_pretrained(net: torch.nn.Module, basenet: str, save_folder: str,
                    model: str, local_rank: int) -> None:
    path = os.path.join(save_folder, basenet)
    if not os.path.isfile(path):
        if local_rank == 0:
            print(f'[WARN] base weights not found at {path} — '
                  f'training backbone from scratch')
        return
    base_weights = torch.load(path)
    if local_rank == 0:
        print(f'Load base network {path}')
    if model in ('vgg', 'dark'):
        net.vgg.load_state_dict(base_weights)
    else:
        net.resnet.load_state_dict(base_weights)


def init_random_layers(net: torch.nn.Module) -> None:
    for layer in (
        net.extras, net.fpn_topdown, net.fpn_latlayer, net.fpn_fem,
        net.loc_pal1, net.conf_pal1, net.loc_pal2, net.conf_pal2, net.ref,
    ):
        layer.apply(net.weights_init)


def build_param_groups(dsfd_net: torch.nn.Module,
                       lr: float) -> List[Dict[str, Any]]:
    main_groups = [
        dsfd_net.vgg, dsfd_net.extras, dsfd_net.fpn_topdown,
        dsfd_net.fpn_latlayer, dsfd_net.fpn_fem,
        dsfd_net.loc_pal1, dsfd_net.conf_pal1,
        dsfd_net.loc_pal2, dsfd_net.conf_pal2,
    ]
    groups = [{'params': m.parameters(), 'lr': lr} for m in main_groups]
    groups.append({'params': dsfd_net.ref.parameters(), 'lr': lr / 10.0})
    return groups


def history_factory() -> History:
    return {
        'total': [], 'pal1_loc': [], 'pal1_conf': [],
        'pal2_loc': [], 'pal2_conf': [],
        'enhance': [], 'enhance_l1ssim': [], 'mutual': [],
        'train_loss_epoch': [], 'val_loss': [],
        'val_accuracy': [], 'val_precision': [], 'val_recall': [],
        'val_f1': [], 'val_map': [],
    }


def record_iter_losses(history: History, iteration: int, tloss: float,
                       loss_l_pa1l: torch.Tensor,
                       loss_c_pal1: torch.Tensor,
                       loss_l_pa12: torch.Tensor,
                       loss_c_pal2: torch.Tensor,
                       loss_enhance: torch.Tensor,
                       loss_enhance2: torch.Tensor,
                       loss_mutual: torch.Tensor) -> None:
    history['total'].append((iteration, float(tloss)))
    history['pal1_loc'].append((iteration, float(loss_l_pa1l.item())))
    history['pal1_conf'].append((iteration, float(loss_c_pal1.item())))
    history['pal2_loc'].append((iteration, float(loss_l_pa12.item())))
    history['pal2_conf'].append((iteration, float(loss_c_pal2.item())))
    history['enhance'].append((iteration, float(loss_enhance.item())))
    history['enhance_l1ssim'].append((iteration, float(loss_enhance2.item())))
    history['mutual'].append((iteration, float(loss_mutual.item())))


def viz_method() -> str:
    return 'DAI-Net (railway, real-target dark)'


def viz_config(args_ns: argparse.Namespace,
               extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        'backbone': args_ns.model,
        'exp': args_ns.num_exp,
        'batch': args_ns.batch_size,
        'nc': args_ns.nc,
        'dark_src': 'synthetic',
    }
    if extra:
        out.update(extra)
    return out


def _inner_net(net: torch.nn.Module) -> torch.nn.Module:
    return net.module if hasattr(net, 'module') else net


def _ensure_detect(net: torch.nn.Module) -> Tuple[Any, torch.nn.Module]:
    """Lazily attach a Detect+Softmax pair to the model for eval-time decoding.

    The model is built with phase='train', so `test_forward` returns the raw
    6-tuple (loc1, conf1, priors1, loc2, conf2, priors2). We cache a Detect
    instance on the inner module so we can decode that tuple into a
    (B, num_classes, top_k, 5) detection tensor.
    """
    from layers.functions.detection import Detect

    inner = _inner_net(net)
    if not hasattr(inner, '_eval_detect'):
        inner._eval_detect = Detect(cfg)
        inner._eval_softmax = torch.nn.Softmax(dim=-1)
    return inner._eval_detect, inner._eval_softmax


def _decode_predictions(net: torch.nn.Module,
                        out_tuple: Tuple[torch.Tensor, ...],
                        ) -> torch.Tensor:
    """Turn the train-phase 6-tuple into a (B, num_classes, top_k, 5) det tensor."""
    detect, softmax = _ensure_detect(net)
    loc_pal2 = out_tuple[3]
    conf_pal2 = out_tuple[4]
    priors_pal2 = out_tuple[5]
    softmax_conf = softmax(conf_pal2)
    # priors come from PriorBox.forward() on CPU; type_as migrates both
    # dtype and device to match loc_pal2.
    return detect.forward(
        loc_pal2,
        softmax_conf,
        priors_pal2.type_as(loc_pal2),
    )


def infer_detections(net: torch.nn.Module, image_chw_01: torch.Tensor,
                     conf_thr: float = 0.05,
                     ) -> Tuple[np.ndarray, np.ndarray]:
    with torch.no_grad():
        x = image_chw_01.unsqueeze(0).cuda()
        forward = (
            net.module.test_forward if hasattr(net, 'module')
            else net.test_forward
        )
        out, _ = forward(x)
        if isinstance(out, tuple):
            # Model was built with phase='train' — decode the raw 6-tuple.
            out = _decode_predictions(net, out)
        det = out.data.cpu().numpy()
    h, w = image_chw_01.shape[1], image_chw_01.shape[2]
    scale = np.array([w, h, w, h], dtype=np.float32)
    boxes: List[List[float]] = []
    scores: List[float] = []
    for c in range(1, det.shape[1]):
        for k in range(det.shape[2]):
            s = float(det[0, c, k, 0])
            if s < conf_thr:
                break
            boxes.append((det[0, c, k, 1:] * scale).tolist())
            scores.append(s)
    return (
        np.asarray(boxes, dtype=np.float32),
        np.asarray(scores, dtype=np.float32),
    )


def collect_target_samples(net: torch.nn.Module, target_folder: str,
                           n_show: int) -> List[Dict[str, Any]]:
    if not os.path.isdir(target_folder):
        return []
    cand = sorted(glob.glob(os.path.join(target_folder, '*')))
    cand = [p for p in cand
            if p.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
    out: List[Dict[str, Any]] = []
    for path in cand[:n_show]:
        img = (Image.open(path)
               .convert('RGB')
               .resize((cfg.INPUT_SIZE, cfg.INPUT_SIZE), Image.BILINEAR))
        arr = np.asarray(img, dtype=np.float32).transpose(2, 0, 1) / 255.0
        tensor = torch.from_numpy(arr).float().cuda()
        boxes, scores = infer_detections(net, tensor)
        out.append({
            'image': np.asarray(img).astype(np.uint8),
            'boxes': boxes, 'scores': scores,
            'labels': ['cls'] * len(boxes),
            'title': os.path.basename(path),
        })
    return out


class TrainingContext:
    def __init__(self, args_ns: argparse.Namespace, local_rank: int) -> None:
        self.args = args_ns
        self.local_rank = local_rank
        self.architecture, self.backbone = resolve_arch_and_backbone(args_ns)
        args_ns.architecture = self.architecture  # expose resolved name
        self.save_folder = os.path.join(
            args_ns.save_folder, self.architecture, self.backbone,
        )
        os.makedirs(self.save_folder, exist_ok=True)
        self.charts_dir = viz.make_charts_dir(
            args_ns.charts_dir, 'train',
            self.architecture, self.backbone, args_ns.num_exp,
        )
        self.train_records_path, self.val_records_path = _records_paths(
            getattr(args_ns, 'records_dir', './records'),
            self.architecture, self.backbone, args_ns.num_exp,
        )
        (
            self.train_dataset,
            self.train_loader,
            self.val_dataset,
            self.val_loader,
        ) = build_data_loaders(args_ns)
        if local_rank == 0:
            print(
                f'Source train: {len(self.train_dataset)} | '
                f'Source val: {len(self.val_dataset)} | '
                f'target_folder (viz only): {args_ns.target_folder}'
            )
        self.history: History = history_factory()
        self.min_loss = float('inf')
        self.best_f1 = -1.0


def update_loss_plot(ctx: TrainingContext, extra: Dict[str, Any]) -> None:
    if ctx.local_rank != 0:
        return
    try:
        viz.plot_losses(
            ctx.history, ctx.charts_dir,
            method=viz_method(),
            config=viz_config(ctx.args, extra),
        )
    except Exception as e:
        print(f'[viz] loss plot failed: {e}')


def run_full_visualisation(ctx: TrainingContext, net: torch.nn.Module,
                           extra: Dict[str, Any]) -> None:
    if ctx.local_rank != 0:
        return
    method = viz_method()
    config = viz_config(ctx.args, extra)

    viz.plot_losses(ctx.history, ctx.charts_dir,
                    method=method, config=config)
    viz.plot_train_vs_val(
        ctx.history.get('train_loss_epoch', []),
        ctx.history.get('val_loss', []),
        ctx.charts_dir, method=method, config=config,
    )

    net.eval()
    n_show = max(1, ctx.args.viz_num_samples)
    per_image: List[Dict[str, Any]] = []
    day_samples: List[Dict[str, Any]] = []
    synth_night_samples: List[Dict[str, Any]] = []
    max_eval_batches = 30

    with torch.no_grad():
        for b_idx, (images, targets, img_paths) in enumerate(ctx.val_loader):
            if b_idx >= max_eval_batches:
                break
            images = images.cuda() / 255.0
            img_dark = build_dark_batch(images)
            for i in range(images.shape[0]):
                pb, ps = infer_detections(net, img_dark[i])
                h_, w_ = img_dark.shape[2], img_dark.shape[3]
                gt = (targets[i].cpu().numpy()
                      if hasattr(targets[i], 'cpu')
                      else np.asarray(targets[i]))
                if gt.size:
                    gt_px = gt[:, :4].copy()
                    gt_px[:, 0] *= w_; gt_px[:, 2] *= w_
                    gt_px[:, 1] *= h_; gt_px[:, 3] *= h_
                else:
                    gt_px = np.zeros((0, 4), dtype=np.float32)
                per_image.append(dict(
                    pred_boxes=pb, pred_scores=ps, gt_boxes=gt_px,
                ))
                if len(day_samples) < n_show:
                    pb_d, ps_d = infer_detections(net, images[i])
                    day_samples.append(dict(
                        image=(images[i].detach().cpu().numpy()
                               .transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8),
                        boxes=pb_d, scores=ps_d,
                        labels=['cls'] * len(pb_d),
                        title=os.path.basename(img_paths[i])
                              if i < len(img_paths) else '',
                    ))
                if len(synth_night_samples) < n_show:
                    synth_night_samples.append(dict(
                        image=(img_dark[i].detach().cpu().numpy()
                               .transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8),
                        boxes=pb, scores=ps,
                        labels=['cls'] * len(pb),
                        title='synth/' + (
                            os.path.basename(img_paths[i])
                            if i < len(img_paths) else ''
                        ),
                    ))

    scores, matched, n_gt, cm = viz.evaluate_detections(
        per_image, iou_thr=0.5, score_thr_cm=0.5,
    )
    viz.plot_confusion_matrix(cm, ctx.charts_dir,
                              method=method, config=config,
                              classes=('background', 'object'))
    viz.plot_pr_curve(scores, matched, n_gt, ctx.charts_dir,
                      method=method, config=config)
    viz.plot_recall_f1_curve(scores, matched, n_gt, ctx.charts_dir,
                             method=method, config=config)
    viz.plot_sample_predictions(
        day_samples, ctx.charts_dir, 'samples_day.png',
        method=method, config=config, title_suffix='(source val / day)',
    )
    viz.plot_sample_predictions(
        synth_night_samples, ctx.charts_dir, 'samples_synth_night.png',
        method=method, config=config,
        title_suffix='(source val + Dark ISP)',
    )

    real_night_samples = collect_target_samples(
        net, ctx.args.target_folder, n_show,
    )
    if real_night_samples:
        viz.plot_sample_predictions(
            real_night_samples, ctx.charts_dir, 'samples_real_night.png',
            method=method, config=config,
            title_suffix='(real target / night)',
        )

    with open(os.path.join(ctx.charts_dir, 'history.json'), 'w') as fh:
        json.dump(dict(ctx.history), fh, indent=2)

    print(f'[viz] saved charts to {ctx.charts_dir}')
    net.train()


def _detect_metrics_from_cm(cm: np.ndarray) -> Dict[str, float]:
    """cm laid out as [[0, FP], [FN, TP]] (see viz.evaluate_detections)."""
    tp = int(cm[1, 1])
    fp = int(cm[0, 1])
    fn = int(cm[1, 0])
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = (2 * precision * recall) / max(precision + recall, 1e-9)
    # Jaccard-style "accuracy" for detection: TP / (TP + FP + FN).
    accuracy = tp / max(tp + fp + fn, 1)
    return dict(tp=tp, fp=fp, fn=fn,
                precision=precision, recall=recall, f1=f1, accuracy=accuracy)


def _format_val_table(epoch: int, rows: List[Tuple[str, str]]) -> str:
    label_w = max(len(k) for k, _ in rows)
    value_w = max(len(v) for _, v in rows)
    width = label_w + value_w + 7
    bar = '+' + '-' * (width - 2) + '+'
    title = f' Val metrics — epoch {epoch} '.center(width - 2, ' ')
    lines = [bar, '|' + title + '|', bar]
    for k, v in rows:
        lines.append(f'| {k:<{label_w}s} : {v:>{value_w}s} |')
    lines.append(bar)
    return '\n'.join(lines)


def _decode_per_image(out_tuple: Tuple[torch.Tensor, ...],
                      targets: Sequence[torch.Tensor],
                      net: torch.nn.Module,
                      conf_thr: float = 0.05,
                      ) -> List[Dict[str, np.ndarray]]:
    """Decode pal2 outputs into per-image (pred_boxes, pred_scores, gt_boxes).

    All boxes are kept in normalised [0,1] coords — IoU is scale-invariant.
    """
    det = _decode_predictions(net, out_tuple).cpu().numpy()
    out: List[Dict[str, np.ndarray]] = []
    for b in range(det.shape[0]):
        boxes: List[List[float]] = []
        scores: List[float] = []
        for cls_id in range(1, det.shape[1]):
            for k in range(det.shape[2]):
                s = float(det[b, cls_id, k, 0])
                if s < conf_thr:
                    break
                boxes.append(det[b, cls_id, k, 1:].tolist())
                scores.append(s)
        gt = (targets[b].cpu().numpy()
              if hasattr(targets[b], 'cpu') else np.asarray(targets[b]))
        gt_boxes = (gt[:, :4].astype(np.float32)
                    if gt.size else np.zeros((0, 4), dtype=np.float32))
        out.append({
            'pred_boxes': (np.asarray(boxes, dtype=np.float32)
                           if boxes else np.zeros((0, 4), dtype=np.float32)),
            'pred_scores': np.asarray(scores, dtype=np.float32),
            'gt_boxes': gt_boxes,
        })
    return out


def validate(ctx: TrainingContext, epoch: int,
             net: torch.nn.Module, dsfd_net: torch.nn.Module,
             net_enh: torch.nn.Module,
             criterion: MultiBoxLoss) -> Optional[float]:
    """Validation: loss + detection metrics on source val + Dark ISP."""
    net.eval()
    net_enh.eval()
    t0 = time.time()
    losses = torch.tensor(0.0, device='cuda')
    loc_sum = torch.tensor(0.0, device='cuda')
    conf_sum = torch.tensor(0.0, device='cuda')
    step = 0
    per_image: List[Dict[str, np.ndarray]] = []

    is_rank0 = ctx.local_rank == 0
    pbar = tqdm(
        ctx.val_loader,
        total=len(ctx.val_loader),
        desc=f'Epoch {epoch} [val]',
        leave=False,
        position=1,
        dynamic_ncols=True,
        disable=not is_rank0,
        unit='batch',
        colour='green',
    )

    test_forward = (net.module.test_forward
                    if hasattr(net, 'module') else net.test_forward)
    with torch.no_grad():
        for images, targets, _ in pbar:
            images = images.cuda() / 255.0
            targets_v = [t.cuda() for t in targets]
            img_dark = build_dark_batch(images)
            out, _ = test_forward(img_dark)
            loss_l_pa12, loss_c_pal2 = criterion(out[3:], targets_v)
            batch_loss = (loss_l_pa12 + loss_c_pal2).detach()
            losses += batch_loss
            loc_sum += loss_l_pa12.detach()
            conf_sum += loss_c_pal2.detach()
            step += 1

            # Detection-metric accumulation (per-rank, used only by rank 0).
            if is_rank0:
                per_image.extend(_decode_per_image(out, targets, net))

            if is_rank0:
                pbar.set_postfix({
                    'loss': f'{batch_loss.item():.3f}',
                    'avg': f'{(losses / step).item():.3f}',
                    'p2_c': f'{loss_c_pal2.item():.3f}',
                    'p2_l': f'{loss_l_pa12.item():.3f}',
                })

    pbar.close()

    dist.reduce(losses, 0, op=dist.ReduceOp.SUM)
    dist.reduce(loc_sum, 0, op=dist.ReduceOp.SUM)
    dist.reduce(conf_sum, 0, op=dist.ReduceOp.SUM)
    n_gpus = max(torch.cuda.device_count(), 1)
    denom = max(step, 1) * n_gpus
    val_loss = (losses / denom).item()
    val_loc = (loc_sum / denom).item()
    val_conf = (conf_sum / denom).item()

    if not is_rank0:
        net.train()
        return None

    # Detection metrics on rank 0's val slice.
    scores, matched, n_gt, cm = viz.evaluate_detections(
        per_image, iou_thr=0.5, score_thr_cm=0.5,
    )
    m = _detect_metrics_from_cm(cm)
    _, _, _, mAP = viz._pr_from_scores(
        np.asarray(scores), np.asarray(matched), n_gt,
    )

    elapsed = time.time() - t0
    table = _format_val_table(epoch, [
        ('val_loss',  f'{val_loss:.4f}'),
        ('pal2_loc',  f'{val_loc:.4f}'),
        ('pal2_conf', f'{val_conf:.4f}'),
        ('accuracy',  f'{m["accuracy"]:.4f}'),
        ('precision', f'{m["precision"]:.4f}'),
        ('recall',    f'{m["recall"]:.4f}'),
        ('f1',        f'{m["f1"]:.4f}'),
        ('mAP@0.5',   f'{mAP:.4f}'),
        ('TP/FP/FN',  f'{m["tp"]}/{m["fp"]}/{m["fn"]}'),
        ('time(s)',   f'{elapsed:.2f}'),
    ])
    print(table)

    ctx.history['val_accuracy'].append((epoch, float(m['accuracy'])))
    ctx.history['val_precision'].append((epoch, float(m['precision'])))
    ctx.history['val_recall'].append((epoch, float(m['recall'])))
    ctx.history['val_f1'].append((epoch, float(m['f1'])))
    ctx.history['val_map'].append((epoch, float(mAP)))

    _append_record_row(
        ctx.val_records_path, _VAL_COLUMNS,
        {
            'epoch': epoch,
            'loss':      f'{val_loss:.6f}',
            'pal2_loc':  f'{val_loc:.6f}',
            'pal2_conf': f'{val_conf:.6f}',
            'accuracy':  f'{m["accuracy"]:.6f}',
            'precision': f'{m["precision"]:.6f}',
            'recall':    f'{m["recall"]:.6f}',
            'f1':        f'{m["f1"]:.6f}',
            'mAP':       f'{float(mAP):.6f}',
            'tp': m['tp'], 'fp': m['fp'], 'fn': m['fn'],
            'elapsed_s': f'{elapsed:.2f}',
            'timestamp': _dt.datetime.now().isoformat(timespec='seconds'),
        },
    )

    if m['f1'] > ctx.best_f1:
        print(f'[ckpt] saving best_model.pth, epoch {epoch} '
              f'(F1 {m["f1"]:.4f} > {max(ctx.best_f1, 0.0):.4f})')
        torch.save(dsfd_net.state_dict(),
                   os.path.join(ctx.save_folder, CHECKPOINT_BEST))
        ctx.best_f1 = m['f1']
    if val_loss < ctx.min_loss:
        ctx.min_loss = val_loss

    torch.save(
        {'epoch': epoch, 'weight': dsfd_net.state_dict()},
        os.path.join(ctx.save_folder, CHECKPOINT_LATEST),
    )
    net.train()
    return val_loss


def train_one_epoch(
    ctx: TrainingContext,
    net: torch.nn.Module,
    net_enh: torch.nn.Module,
    criterion: MultiBoxLoss,
    criterion_enh: EnhanceLoss,
    optimizer: optim.Optimizer,
    epoch: int, iteration: int, step_index: int,
) -> Tuple[int, int]:
    
    losses_sum = 0.0

    comp_sums = {
        'pal1_loc': 0.0, 'pal1_conf': 0.0,
        'pal2_loc': 0.0, 'pal2_conf': 0.0,
        'enhance': 0.0, 'enhance_l1ssim': 0.0, 'mutual': 0.0,
    }
    epoch_start = time.time()
    batch_idx = 0
    is_rank0 = ctx.local_rank == 0
    n_batches_est = len(ctx.train_loader)
    pbar = tqdm(
        ctx.train_loader,
        total=n_batches_est,
        desc=f'Epoch {epoch} [train]',
        leave=False,
        position=1,
        dynamic_ncols=True,
        disable=not is_rank0,
        unit='batch',
        colour='green',
    )

    for batch_idx, (images, targets, _) in enumerate(pbar):
        images = Variable(images.cuda() / 255.0)
        targets_v = [
            Variable(ann.cuda(), requires_grad=False) for ann in targets
        ]
        img_dark = build_dark_batch(images)

        if iteration in cfg.LR_STEPS:
            step_index += 1
            adjust_learning_rate(optimizer, ctx.args.gamma)

        t0 = time.time()
        R_dark_gt, I_dark = net_enh(img_dark)
        R_light_gt, I_light = net_enh(images)

        out, out2, loss_mutual = net(
            img_dark, images, I_dark.detach(), I_light.detach(),
        )
        R_dark, R_light, R_dark_2, R_light_2 = out2

        optimizer.zero_grad()
        loss_l_pa1l, loss_c_pal1 = criterion(out[:3], targets_v)
        loss_l_pa12, loss_c_pal2 = criterion(out[3:], targets_v)

        loss_enhance = criterion_enh(
            [R_dark, R_light, R_dark_2, R_light_2,
             I_dark.detach(), I_light.detach()],
            images, img_dark,
        ) * 0.1
        loss_enhance2 = (
            F.l1_loss(R_dark, R_dark_gt.detach())
            + F.l1_loss(R_light, R_light_gt.detach())
            + (1.0 - ssim(R_dark, R_dark_gt.detach()))
            + (1.0 - ssim(R_light, R_light_gt.detach()))
        )
        loss = (
            loss_l_pa1l + loss_c_pal1 + loss_l_pa12 + loss_c_pal2
            + loss_enhance2 + loss_enhance + loss_mutual
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            net.parameters(), max_norm=35, norm_type=2,
        )
        optimizer.step()
        t1 = time.time()
        losses_sum += loss.item()
        if is_rank0:
            comp_sums['pal1_loc']        += float(loss_l_pa1l.item())
            comp_sums['pal1_conf']       += float(loss_c_pal1.item())
            comp_sums['pal2_loc']        += float(loss_l_pa12.item())
            comp_sums['pal2_conf']       += float(loss_c_pal2.item())
            comp_sums['enhance']         += float(loss_enhance.item())
            comp_sums['enhance_l1ssim']  += float(loss_enhance2.item())
            comp_sums['mutual']          += float(loss_mutual.item())

        if is_rank0:
            cur_lr = optimizer.param_groups[0]['lr']
            tloss_running = losses_sum / (batch_idx + 1)
            pbar.set_postfix({
                'loss': f'{loss.item():.3f}',
                'avg': f'{tloss_running:.3f}',
                'p1_c': f'{loss_c_pal1.item():.3f}',
                'p1_l': f'{loss_l_pa1l.item():.3f}',
                'p2_c': f'{loss_c_pal2.item():.3f}',
                'p2_l': f'{loss_l_pa12.item():.3f}',
                'enh': f'{loss_enhance.item():.3f}',
                'enh2': f'{loss_enhance2.item():.3f}',
                'mut': f'{loss_mutual.item():.3f}',
                'lr': f'{cur_lr:.2e}',
                'it': iteration,
                'dt': f'{t1 - t0:.2f}s',
            })

        if iteration % PRINT_EVERY == 0 and is_rank0:
            tloss = losses_sum / (batch_idx + 1)
            cur_lr = optimizer.param_groups[0]['lr']
            tqdm.write(
                f'[train] ep:{epoch} it:{iteration} '
                f'loss(avg):{tloss:.4f} '
                f'p1[c:{loss_c_pal1.item():.4f} l:{loss_l_pa1l.item():.4f}] '
                f'p2[c:{loss_c_pal2.item():.4f} l:{loss_l_pa12.item():.4f}] '
                f'enh:{loss_enhance.item():.4f} '
                f'enh2:{loss_enhance2.item():.4f} '
                f'mut:{loss_mutual.item():.4f} '
                f'lr:{cur_lr:.2e} dt:{t1 - t0:.3f}s'
            )
            record_iter_losses(
                ctx.history, iteration, tloss,
                loss_l_pa1l, loss_c_pal1, loss_l_pa12, loss_c_pal2,
                loss_enhance, loss_enhance2, loss_mutual,
            )

        if (is_rank0
                and ctx.args.viz_every_iters > 0
                and iteration > 0
                and iteration % ctx.args.viz_every_iters == 0):
            update_loss_plot(ctx, {'iter': iteration, 'epoch': epoch})

        iteration += 1

    pbar.close()
    if is_rank0:
        n_batches = max(1, batch_idx + 1)
        train_mean_loss = float(losses_sum / n_batches)
        ctx.history['train_loss_epoch'].append((epoch, train_mean_loss))
        _append_record_row(
            ctx.train_records_path, _TRAIN_COLUMNS,
            {
                'epoch': epoch,
                'iteration': iteration,
                'lr': f'{optimizer.param_groups[0]["lr"]:.6e}',
                'loss': f'{train_mean_loss:.6f}',
                'pal1_loc':       f'{comp_sums["pal1_loc"]       / n_batches:.6f}',
                'pal1_conf':      f'{comp_sums["pal1_conf"]      / n_batches:.6f}',
                'pal2_loc':       f'{comp_sums["pal2_loc"]       / n_batches:.6f}',
                'pal2_conf':      f'{comp_sums["pal2_conf"]      / n_batches:.6f}',
                'enhance':        f'{comp_sums["enhance"]        / n_batches:.6f}',
                'enhance_l1ssim': f'{comp_sums["enhance_l1ssim"] / n_batches:.6f}',
                'mutual':         f'{comp_sums["mutual"]         / n_batches:.6f}',
                'elapsed_s': f'{time.time() - epoch_start:.2f}',
                'timestamp': _dt.datetime.now().isoformat(timespec='seconds'),
            },
        )
    return iteration, step_index


def train(ctx: TrainingContext) -> None:
    args_ns = ctx.args
    n_gpus = max(torch.cuda.device_count(), 1)
    per_epoch_size = len(ctx.train_dataset) // (args_ns.batch_size * n_gpus)

    basenet = basenet_factory(args_ns.model)
    num_classes = args_ns.nc + 1  # +1 for background

    cfg.NUM_CLASSES = num_classes
    # Schedule overrides from YAML (epochs/max_steps/lr_steps). These must be
    # applied before the LR-step loop and the epoch loop consult cfg.
    cfg.EPOCHES = int(args_ns.epochs)
    cfg.MAX_STEPS = int(args_ns.max_steps)
    if args_ns.lr_steps:
        cfg.LR_STEPS = tuple(int(s) for s in args_ns.lr_steps)

    dsfd_net = build_net('train', num_classes, args_ns.model)
    net = dsfd_net
    net_enh = RetinexNet()
    retinex_path = os.path.join(args_ns.save_folder, RETINEX_WEIGHTS)
    if os.path.isfile(retinex_path):
        net_enh.load_state_dict(torch.load(retinex_path))
        if ctx.local_rank == 0:
            print(f'Loaded RetinexNet from {retinex_path}')
    elif ctx.local_rank == 0:
        print(f'[WARN] {retinex_path} missing — RetinexNet trained from scratch '
              f'(pseudo-GT will be noisy)')

    start_epoch = 0
    iteration = 0
    if args_ns.resume:
        if ctx.local_rank == 0:
            print(f'Resuming training, loading {args_ns.resume}...')
        start_epoch = net.load_weights(args_ns.resume)
        iteration = start_epoch * per_epoch_size
    else:
        load_pretrained(net, basenet, args_ns.save_folder,
                        args_ns.model, ctx.local_rank)
        if ctx.local_rank == 0:
            print('Initializing weights...')
        init_random_layers(net)

    lr = args_ns.lr * np.round(
        np.sqrt(args_ns.batch_size / 4 * n_gpus), 4,
    )
    optimizer = optim.SGD(
        build_param_groups(dsfd_net, lr),
        lr=lr, momentum=args_ns.momentum, weight_decay=args_ns.weight_decay,
    )

    if args_ns.cuda and args_ns.multigpu:
        net = torch.nn.parallel.DistributedDataParallel(
            net.cuda(), find_unused_parameters=False,
        )
        net_enh = torch.nn.parallel.DistributedDataParallel(net_enh.cuda())
        cudnn.benchmark = True

    criterion = MultiBoxLoss(cfg, args_ns.cuda)
    criterion_enh = EnhanceLoss()

    if ctx.local_rank == 0:
        print('Using the specified args:')
        print(args_ns)
        print(f'Charts dir: {ctx.charts_dir}')
        print(f'Num classes: {num_classes} (= {args_ns.nc} fg + 1 bg)')

    step_index = 0
    for step in cfg.LR_STEPS:
        if iteration > step:
            step_index += 1
            adjust_learning_rate(optimizer, args_ns.gamma)
    net_enh.eval()
    net.train()

    is_rank0 = ctx.local_rank == 0
    epoch_pbar = tqdm(
        range(start_epoch, cfg.EPOCHES),
        desc='Epochs',
        total=cfg.EPOCHES - start_epoch,
        leave=True,
        position=0,
        dynamic_ncols=True,
        disable=not is_rank0,
        unit='ep',
        colour='cyan',
    )

    epoch = start_epoch
    for epoch in epoch_pbar:
        iteration, step_index = train_one_epoch(
            ctx, net, net_enh,
            criterion, criterion_enh, optimizer,
            epoch, iteration, step_index,
        )
        val_loss = validate(ctx, epoch, net, dsfd_net, net_enh, criterion)
        if is_rank0 and val_loss is not None:
            ctx.history['val_loss'].append((epoch, float(val_loss)))
            epoch_pbar.set_postfix({
                'val': f'{val_loss:.4f}',
                'best': f'{ctx.min_loss:.4f}',
                'lr': f'{optimizer.param_groups[0]["lr"]:.2e}',
                'iter': iteration,
            })

        if (is_rank0
                and args_ns.viz_full_every_epochs > 0
                and (epoch + 1) % args_ns.viz_full_every_epochs == 0):
            try:
                run_full_visualisation(
                    ctx, net,
                    {'lr': lr, 'epoch': epoch + 1, 'iter': iteration},
                )
            except Exception as e:
                print(f'[WARN] periodic visualisation failed: {e}')

        if iteration >= cfg.MAX_STEPS:
            break

    epoch_pbar.close()

    if ctx.local_rank == 0:
        try:
            run_full_visualisation(
                ctx, net,
                {'lr': lr, 'epochs': epoch + 1, 'iter': iteration},
            )
        except Exception as e:
            print(f'[WARN] final visualisation failed: {e}')


def main() -> None:
    args_ns = parse_args()
    local_rank = args_ns.local_rank
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(local_rank)
    if local_rank == 0:
        setup_logging(args_ns.model, args_ns.num_exp, args_ns)
    setup_distributed(local_rank, args_ns.cuda)
    ctx = TrainingContext(args_ns, local_rank)
    train(ctx)


if __name__ == '__main__':
    main()
