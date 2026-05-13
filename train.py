"""DAI-Net training entry point — DSFD backbone, paired Dark ISP, YOLO dataset.

Source domain: user's labelled well-lit images (YOLO format) — used both for
detection supervision and as the input to Dark ISP for producing paired dark
samples (paper-faithful interchange-redecomposition-coherence procedure).

Target domain: user's real low-light video frames — consumed only for
end-of-epoch sample visualisation (no labels available).
"""
from __future__ import annotations

import argparse
import datetime as _dt
import glob
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
from PIL import Image
from torch.autograd import Variable
from torchmetrics.functional import structural_similarity_index_measure as ssim

from data.config import cfg
from data.widerface import WIDERDetection, detection_collate
from layers.modules import EnhanceLoss, MultiBoxLoss
from models.enhancer import RetinexNet
from models.factory import basenet_factory, build_net
from utils import visualize as viz
from utils.dark_isp import Low_Illumination_Degrading


Point = Tuple[float, float]
History = Dict[str, List[Point]]


CHECKPOINT_LATEST = 'dsfd_checkpoint.pth'
CHECKPOINT_BEST = 'dsfd.pth'
ITER_CHECKPOINT_FMT = 'dsfd_{iteration}.pth'
RETINEX_WEIGHTS = 'decomp.pth'
PRINT_EVERY = 100
ITER_CKPT_EVERY = 5000


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser('DSFD training on a YOLO-format dataset (DAI-Net)')
    p.add_argument('--batch_size', default=4, type=int)
    p.add_argument(
        '--model', default='dark', type=str,
        choices=['dark', 'vgg', 'resnet50', 'resnet101', 'resnet152'],
    )
    p.add_argument('--resume', default=None, type=str)
    p.add_argument('--num_workers', default=0, type=int)
    p.add_argument('--cuda', default=True, type=bool)
    p.add_argument('--lr', '--learning-rate', default=5e-4, type=float)
    p.add_argument('--momentum', default=0.9, type=float)
    p.add_argument('--weight_decay', default=5e-4, type=float)
    p.add_argument('--gamma', default=0.1, type=float)
    p.add_argument('--multigpu', default=True, type=bool)
    p.add_argument('--save_folder', default='weights/', type=str)
    p.add_argument('--local_rank', type=int, default=0)
    p.add_argument('--train_file', default='./dataset/source_train.txt',
                   type=str,
                   help='Path to DAI-Net format train txt '
                        '(produced by convert_yolo_to_dainet.py)')
    p.add_argument('--val_file', default='./dataset/source_val.txt',
                   type=str)
    p.add_argument('--nc', default=3, type=int,
                   help='Number of foreground classes (background is added)')
    p.add_argument(
        '--target_folder',
        default='/media/caotulab/303A225B3A221DFA/Nhan/data/images/target',
        type=str,
        help='Folder with real target-domain (dark) images for visualisation',
    )
    p.add_argument('--num_exp', default='exp1', type=str)
    p.add_argument('--charts_dir', default='./charts', type=str)
    p.add_argument('--viz_num_samples', default=6, type=int)
    p.add_argument('--viz_every_iters', default=500, type=int)
    p.add_argument('--viz_full_every_epochs', default=1, type=int)
    return p.parse_args()


class Tee:
    """Mirror writes to a stream and a file."""

    def __init__(self, stream: Any, file_handle: Any) -> None:
        self._stream = stream
        self._file = file_handle

    def write(self, data: str) -> None:
        self._stream.write(data)
        try:
            self._file.write(data)
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
) -> Tuple[WIDERDetection, data.DataLoader,
           WIDERDetection, data.DataLoader]:
    train_ds = WIDERDetection(args_ns.train_file, mode='train')
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

    val_ds = WIDERDetection(args_ns.val_file, mode='val')
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
        'batch': args_ns.batch_size,
        'nc': args_ns.nc,
        'dark_src': 'synthetic',
    }
    if extra:
        out.update(extra)
    return out


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
        self.save_folder = os.path.join(args_ns.save_folder, args_ns.model)
        os.makedirs(self.save_folder, exist_ok=True)
        self.charts_dir = viz.make_charts_dir(
            args_ns.charts_dir, 'train', args_ns.model, args_ns.num_exp,
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


def validate(ctx: TrainingContext, epoch: int,
             net: torch.nn.Module, dsfd_net: torch.nn.Module,
             net_enh: torch.nn.Module,
             criterion: MultiBoxLoss) -> Optional[float]:
    """Validation on the source val set with Dark ISP synth + MultiBoxLoss."""
    net.eval()
    net_enh.eval()
    t0 = time.time()
    losses = torch.tensor(0.0, device='cuda')
    step = 0

    with torch.no_grad():
        for images, targets, _ in ctx.val_loader:
            images = images.cuda() / 255.0
            targets_v = [t.cuda() for t in targets]
            img_dark = build_dark_batch(images)
            out, _ = (net.module.test_forward
                      if hasattr(net, 'module')
                      else net.test_forward)(img_dark)
            loss_l_pa12, loss_c_pal2 = criterion(out[3:], targets_v)
            losses += (loss_l_pa12 + loss_c_pal2).detach()
            step += 1

    dist.reduce(losses, 0, op=dist.ReduceOp.SUM)
    n_gpus = max(torch.cuda.device_count(), 1)
    val_loss = (losses / max(step, 1) / n_gpus).item()

    if ctx.local_rank != 0:
        net.train()
        return None

    print(f'Timer: {time.time() - t0:.4f}')
    print(f'val[source+ISP] epoch:{epoch} | loss:{val_loss:.4f}')

    if val_loss < ctx.min_loss:
        print(f'Saving best state, epoch {epoch}')
        torch.save(dsfd_net.state_dict(),
                   os.path.join(ctx.save_folder, CHECKPOINT_BEST))
        ctx.min_loss = val_loss

    torch.save(
        {'epoch': epoch, 'weight': dsfd_net.state_dict()},
        os.path.join(ctx.save_folder, CHECKPOINT_LATEST),
    )
    net.train()
    return val_loss


def train_one_epoch(ctx: TrainingContext,
                    net: torch.nn.Module,
                    dsfd_net: torch.nn.Module,
                    net_enh: torch.nn.Module,
                    criterion: MultiBoxLoss,
                    criterion_enh: EnhanceLoss,
                    optimizer: optim.Optimizer,
                    epoch: int, iteration: int, step_index: int, lr: float,
                    ) -> Tuple[int, int]:
    losses_sum = 0.0
    batch_idx = 0
    for batch_idx, (images, targets, _) in enumerate(ctx.train_loader):
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

        if iteration % PRINT_EVERY == 0:
            tloss = losses_sum / (batch_idx + 1)
            if ctx.local_rank == 0:
                print(f'Timer: {t1 - t0:.4f}')
                print(f'epoch:{epoch} || iter:{iteration} || Loss:{tloss:.4f}')
                print(f'->> pal1 conf {loss_c_pal1.item():.4f} | '
                      f'pal1 loc {loss_l_pa1l.item():.4f}')
                print(f'->> pal2 conf {loss_c_pal2.item():.4f} | '
                      f'pal2 loc {loss_l_pa12.item():.4f}')
                print(f'->>lr:{optimizer.param_groups[0]["lr"]}')
                record_iter_losses(
                    ctx.history, iteration, tloss,
                    loss_l_pa1l, loss_c_pal1, loss_l_pa12, loss_c_pal2,
                    loss_enhance, loss_enhance2, loss_mutual,
                )

        if (ctx.local_rank == 0
                and ctx.args.viz_every_iters > 0
                and iteration > 0
                and iteration % ctx.args.viz_every_iters == 0):
            update_loss_plot(ctx, {'iter': iteration, 'epoch': epoch})

        if (iteration != 0 and iteration % ITER_CKPT_EVERY == 0
                and ctx.local_rank == 0):
            print(f'Saving state, iter: {iteration}')
            torch.save(
                dsfd_net.state_dict(),
                os.path.join(
                    ctx.save_folder,
                    ITER_CHECKPOINT_FMT.format(iteration=iteration),
                ),
            )
        iteration += 1

    if ctx.local_rank == 0:
        n_batches = max(1, batch_idx + 1)
        ctx.history['train_loss_epoch'].append(
            (epoch, float(losses_sum / n_batches)),
        )
    return iteration, step_index


def train(ctx: TrainingContext) -> None:
    args_ns = ctx.args
    n_gpus = max(torch.cuda.device_count(), 1)
    per_epoch_size = len(ctx.train_dataset) // (args_ns.batch_size * n_gpus)

    basenet = basenet_factory(args_ns.model)
    num_classes = args_ns.nc + 1  # +1 for background
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
            net.cuda(), find_unused_parameters=True,
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

    epoch = start_epoch
    for epoch in range(start_epoch, cfg.EPOCHES):
        iteration, step_index = train_one_epoch(
            ctx, net, dsfd_net, net_enh,
            criterion, criterion_enh, optimizer,
            epoch, iteration, step_index, lr,
        )
        val_loss = validate(ctx, epoch, net, dsfd_net, net_enh, criterion)
        if ctx.local_rank == 0 and val_loss is not None:
            ctx.history['val_loss'].append((epoch, float(val_loss)))

        if (ctx.local_rank == 0
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
