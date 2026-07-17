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
- 🔌 Swappable backbones: `vgg16`, `vgg16_sppf`, `resnet50/101/152`, `yolo26n/s`
- ⚙️ Everything driven by one YAML config — no code edits per experiment
- 🚀 Multi-GPU training via `torchrun` (DDP)
- 📊 Evaluation on source / target / combined with mAP, confusion matrix, Grad-CAM, t-SNE

## 🛠️ Setup

```bash
conda create -n railight python=3.10 -y
conda activate railight

pip install torch torchvision          # match your CUDA driver
pip install -r requirements.txt        # keep numpy<2
```

Convert YOLO-format data to the RAILIGHT list format:

```bash
python src/utils/convert_yolo_to_railight.py \
    --source-root /path/to/source --out-dir ./dataset \
    --out-prefix source --max-class 3

python src/utils/convert_yolo_to_railight.py \
    --source-root /path/to/target --out-dir ./dataset \
    --out-prefix target --max-class 3
```

This writes `dataset/{source,target}_{train,val,test}.txt`, one line per image with pixel boxes `x y w h cls`.

Pretrained weights go under `weights/`:
- `weights/vgg16_reducedfc.pth` — VGG backbone init
- `weights/decomp.pth` — frozen Retinex DecomNet

## 🚀 Usage

Pick or edit a config, then train:

```bash
python train.py --config configs/train/railight/vgg16/exp1.yaml
```

Multi-GPU (`nproc_per_node` **must** equal the number of GPUs in `gpu_ids`):

```bash
torchrun --standalone --nproc_per_node=2 \
    train.py --config configs/train/railight/vgg16/exp1.yaml
```

Or just use the shell wrappers — they read `gpu_ids` from the config and set `CUDA_VISIBLE_DEVICES` for you:

```bash
bash src/scripts/train.sh
bash src/scripts/test.sh
```

Evaluate a checkpoint:

```bash
python test.py --config configs/test/railight/vgg16/exp1.yaml
```

🔑 Config keys worth knowing:

| Key | Meaning |
| --- | --- |
| `architecture` / `backbone` | `railight` or `dsfd` + backbone name |
| `gpu_ids` | GPUs to use, e.g. `0,1` |
| `kl_loss_weight` | strength of source↔target reflectance alignment |
| `target_loss_weight` | unsupervised Retinex loss on night frames |
| `is_use_rc_loss` | redecomposition cohering loss on/off |
| `focal_enabled` | focal loss for class imbalance |
| `use_wandb` | log to Weights & Biases |

📦 Outputs:
- `weights/<arch>/<backbone>/<exp>/best_model.pth`, `last_model.pth`
- `charts/<mode>/<arch>/<backbone>/<exp>/` — `confusion_matrix_*.png`, `samples_day.png`, `samples_real_night.png`, `tsne_reflectance.png`
- `records/` — per-iteration JSONL logs
- `metrics.json` — mAP@0.5, mAP@0.5:0.95, P/R/F1, FPS and params, split into `combined` / `source` / `target`

## 🎯 Conclusion

RAILIGHT shows that a railway defect detector can be trained with day labels only and still work at night 🌙. Synthetic Dark-ISP images provide supervision under night statistics, and reflectance alignment bridges the remaining gap to real footage — no target annotations needed.

🙏 Built on DAI-Net (Du et al., CVPR 2024), DSFD, Ultralytics YOLO, and the model-adaptation regularisers of Li et al., CVPR 2020.
