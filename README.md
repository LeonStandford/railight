<p align="center">
  <h1 align="center">RAILIGHT: A Robust Illumination-Adaptive Framework for Railway Defect Detection
</h1>
  <p align="center">
    <b>Pham Minh Long</b> — National Yang Ming Chiao Tung University (NYCU), Taiwan
  </p>
</p>

![overview](./assets/railight.jpg)

## 🚂 Introduction

Railway defect detectors are trained on well-lit daytime images, but real inspections often run at night. RAILIGHT closes that gap with **unsupervised domain adaptation**: it learns from labelled day images and adapts to **unlabelled** real night frames.

Each training step sees three domains:

| Domain | Data | Labels |
| --- | --- | --- |
| ☀️ Source | well-lit railway images | ✅ |
| 🌗 Synthetic dark | source + Dark-ISP degradation | ✅ (same GT) |
| 🌙 Target | real night video frames | ❌ |

The model is a DSFD-style dual-shot detector with a frozen **Retinex** branch (DAI-Net, CVPR 2024). Reflectance is illumination-invariant, so aligning it across domains makes the detector robust to lighting. Losses include supervised detection, Retinex reconstruction, cross-domain KL alignment, weight anchoring and entropy minimisation.

✨ Highlights:
- 🔌 Backbones: `vgg16`, `vgg16_sppf`, `yolo26n` (`railight`), plus `vgg16`, `resnet50/101/152` (`dsfd`)
- ⚙️ Everything driven by one YAML config — no code edits per experiment
- 🚀 Multi-GPU training via `torchrun` (DDP)
- 📊 Evaluation on source / target with mAP, confusion matrix, Grad-CAM, t-SNE

## 🛠️ Setup

```bash
conda create -n railight python=3.10 -y
conda activate railight

pip install torch torchvision          # match your CUDA driver
pip install -r requirements.txt        # keep numpy<2
```

Convert YOLO-format data to the list format:

```bash
python src/utils/convert_yolo_to_railight.py \
    --source-root /path/to/source --out-dir ./dataset \
    --out-prefix source --max-class 3

python src/utils/convert_yolo_to_railight.py \
    --source-root /path/to/target --out-dir ./dataset \
    --out-prefix target --max-class 3
```

This writes `dataset/{source,target}_{train,val,test}.txt`, one line per image with pixel boxes `x y w h cls`. See `src/scripts/convert_yolo_to_railight.sh` for the flat-folder and `--class-map` options.

Pretrained weights go under `save_folder`:
- `vgg16_reducedfc.pth` — VGG backbone init
- `decomp.pth` — frozen Retinex DecomNet

## 🚀 Usage

Train:

```bash
python train.py --config configs/train/railight/vgg16/exp3.da.batch8.yaml
```

Multi-GPU (`nproc_per_node` **must** equal the number of GPUs in `gpu_ids`):

```bash
torchrun --standalone --nproc_per_node=2 \
    train.py --config configs/train/railight/vgg16/exp3.da.batch8.yaml
```

Shell wrappers read `gpu_ids` from the config and set `CUDA_VISIBLE_DEVICES`; `train.sh` also takes `CONFIG=...`:

```bash
CONFIG=configs/train/railight/vgg16/exp3.da.batch8.yaml bash src/scripts/train.sh
bash src/scripts/test.sh
```

Evaluate a checkpoint:

```bash
python test.py --config configs/test/railight/vgg16/exp3.yaml
```

🔑 Config keys worth knowing:

| Key | Meaning |
| --- | --- |
| `architecture` / `backbone` | `railight` or `dsfd` + backbone name |
| `gpu_ids` | GPUs to use, e.g. `0` or `0,1` |
| `kl_loss_weight` | strength of source↔target reflectance alignment |
| `target_loss_weight` | unsupervised Retinex loss on night frames |
| `is_use_supervised_target_loss` | also train on target labels (no longer pure UDA) |
| `is_use_rc_loss` | redecomposition cohering loss on/off |
| `focal_enabled` / `focal_class_weights` | focal loss for class imbalance |
| `box_loss` | `smooth_l1`, `ciou` or `wiou` |
| `val_every_epochs` / `viz_full_every_epochs` | validation / full-chart frequency |
| `resume` | `true` resumes `last_model.pth` of the same `num_exp` |
| `use_wandb` | log to Weights & Biases |

`RAILIGHT_PROFILE=0` turns off per-stage timing.

📦 Outputs:
- `<save_folder>/<arch>/<backbone>/<num_exp>/best_model.pth`, `last_model.pth`
- `<charts_dir>/<mode>/<arch>/<backbone>/<num_exp>/` — `losses.png`, `confusion_matrix_*.png`, `samples_day.png`, `samples_real_night.png`, `cat_aug_*.png`
- `<records_dir>/` — per-epoch train / val JSONL
- `metrics.json` (test) — P/R/F1/mAP for source and target

## 🧮 New: CAT + weak/strong augmentation

Ported from CAT ([Kennerley et al., CVPR 2024](https://arxiv.org/abs/2403.19278)) to the dense-prior DSFD head:

- **ICRm**: EMA inter-class relation matrix (source/target) from positive priors — `src/losses/cat_icrm.py`
- **CALoss**: relation-weighted classification loss in `MultiBoxLoss`, weights capped by `cat_max_weight`
- **Weak/strong aug**: teacher labels weak target, student learns on strong target, plus a supervised strong source view — `src/utils/strong_aug.py`
- **Viz**: `cat_aug_<class>.png` per minority class; the YAML config is uploaded to wandb as an artifact
- **Fixes**: source loader `drop_last`, unlabelled target frames now RGB
- Not ported yet: CRA mixup, soft labels

Ablation configs (`configs/train/railight/vgg16/`):

| Step | exp3 (labelled target) | exp4 (UDA) |
| --- | --- | --- |
| Baseline | `exp3.da.batch8` | `exp4.uda.batch8` |
| + weak/strong | `exp3.da.batch8.weak_strong_augmentation` | `exp4.uda.batch8.weak_strong_augmentation` |
| + CAT | `exp3.da.batch8.weak_strong_augmentation.cat` | `exp4.uda.batch8.weak_strong_augmentation.cat` |

Submit on Slurm (from `src/scripts/`):

```bash
CONFIG=configs/train/railight/vgg16/<name>.yaml sbatch -J <name> srun.sh
```

| Key | Meaning |
| --- | --- |
| `cat_enabled` | ICRm + CALoss on/off |
| `cat_warmup_iters` | CALoss starts after this many iterations |
| `cat_max_weight` | cap on CALoss weights |
| `pseudo_weight` / `pseudo_start_iters` | mean-teacher pseudo-label loss and its start |
| `source_strong_weight` / `pseudo_strong_aug` | strong view for source / target |
| `strong_aug_erase_p` | random-erasing probabilities (`[]` = off) |

## 🎯 Conclusion

RAILIGHT shows that a railway defect detector can be trained with day labels only and still work at night 🌙. Synthetic Dark-ISP images provide supervision under night statistics, and reflectance alignment bridges the remaining gap to real footage — no target annotations needed.

🙏 Built on DAI-Net (Du et al., CVPR 2024), DSFD, Ultralytics YOLO, and the model-adaptation regularisers of Li et al., CVPR 2020.
