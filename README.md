<p align="center">
  <h1 align="center">DAI-Net for Low-Light Object Detection — Custom Dataset Adaptation</h1>
  <p align="center">
    <b>Pham Minh Long</b><br>
    National Yang Ming Chiao Tung University (NYCU), Taiwan
  </p>
</p>

PyTorch implementation of low-light object detection with zero-shot day-night
domain adaptation, adapted from the CVPR 2024 paper
**"Boosting Object Detection with Zero-Shot Day-Night Domain Adaptation"**
([arXiv:2312.01220](https://arxiv.org/abs/2312.01220)) by Du et al. This
fork extends the original face-detection pipeline so that it can train on
**arbitrary YOLO-format datasets** (railway defects, custom objects, …) with
a real low-light **target domain** sampled from video.

![overview](./assets/overview.png)

---

## ✨ What's new in this fork

| Area | Original (Du et al.) | This fork |
|---|---|---|
| **Detection task** | Face only (WIDER → DARK FACE) | Any YOLO-format multi-class dataset |
| **Dataset format** | WIDER `wider_face_train.txt` | YOLO `.txt` per image → converted on demand |
| **Source domain** | WIDER Face | User-provided well-lit + labelled images |
| **Target domain** | Synthetic Dark ISP only | Real video frames + Dark ISP for paired training |
| **Alternative detector** | DSFD only | DSFD **or** YOLO26 (Ultralytics) |
| **Visualisation** | None | Loss subplots, PR / CM / Recall-F1 curves, day vs night samples, Grad-CAM, train vs val overlay |
| **Tooling** | — | Frame extractor from videos, dataset converter, training/evaluation shell scripts |
| **Logging** | Console only | Tee-mirrored `logs/<datetime>.log` |

---

## 🚀 Installation

```bash
git clone <this-repo>
cd DAI-Net

conda create -y -n dainet python=3.10
conda activate dainet

pip install -r requirements.txt
```

GPU: tested on NVIDIA RTX 4090 (24 GB), Ubuntu 24.04, CUDA 12.x.

---

## 📁 Repository layout

```
DAI-Net/
├── train.py                  # DSFD + DAI-Net training (paired Dark ISP)
├── train.sh                  #   ↳ shell wrapper
├── train_yolo.py             # YOLO26 alternative trainer
├── train_yolo.sh             #   ↳ shell wrapper
├── cut_frames.py             # Extract video frames -> images/target/
├── cut_frames.sh
├── convert_yolo_to_dainet.py # YOLO label format -> DAI-Net txt format
├── data/
│   ├── widerface.py          # WIDER-style dataset loader (used after conversion)
│   ├── target_domain.py      # Real dark-image loader
│   ├── config.py
│   └── ...
├── models/
│   ├── dai_net.py            # DAI-Net (DSFD + reflectance decoder)
│   ├── enhancer.py           # RetinexNet pseudo-GT generator
│   └── factory.py
├── layers/                   # DSFD detection heads, anchors, losses
├── utils/
│   ├── visualize.py          # All chart helpers + Grad-CAM standalone main()
│   ├── dark_isp.py           # Physics-based dark synthesis
│   └── augmentations.py
├── yolo/ultralytics/         # In-repo Ultralytics package (YOLO26 architecture)
├── dataset/                  # Converted txt files end up here
└── logs/                     # Tee-mirrored training logs (auto-created)
```

---

## 📥 Data and weight preparation

### 1. Source dataset (YOLO format)

Organise your labelled well-lit data as:

```
<source-root>/
├── Train/
│   ├── images/                 # *.jpg / *.png
│   └── labels/                 # *.txt — each line: "cls cx cy w h" (xywh normalised)
├── Val/
│   ├── images/
│   └── labels/
└── data.yaml                   # nc, names, …  (optional but recommended)
```

### 2. Convert YOLO → DAI-Net format

```bash
python convert_yolo_to_dainet.py \
    --source-root /path/to/source \
    --train-split Train --val-split Val \
    --out-dir dataset \
    --max-class 3
```

Produces `dataset/source_train.txt` and `dataset/source_val.txt` in the
DAI-Net (WIDER-style) format expected by `data.widerface.WIDERDetection`.

### 3. Target domain (optional — for visualisation / Grad-CAM)

Extract frames from your low-light videos:

```bash
python cut_frames.py \
    --path-input  /path/to/videos \
    --path-output /path/to/target \
    --fps 1.0
```

### 4. Pretrained weights

| File | Drive link | Where |
|---|---|---|
| RetinexNet `decomp.pth` | [link](https://drive.google.com/file/d/1MaRK-VZmjBvkm79E1G77vFccb_9GWrfG/view) | `weights/decomp.pth` |
| VGG16 base `vgg16_reducedfc.pth` | [link](https://drive.google.com/file/d/1whV71K42YYduOPjTTljBL8CB-Qs4Np6U/view) | `weights/vgg16_reducedfc.pth` |

If either is missing the training script falls back to scratch / no-pseudo-GT
with a warning, but quality will suffer.

---

## 🏋️ Training

### DSFD + DAI-Net (paper-faithful)

```bash
bash train.sh                           # defaults: backbone=dark, nc=3, batch=1
NC=3 BATCH_SIZE=2 ./train.sh            # tweak knobs via env
BACKBONE=resnet50 NUM_EXP=exp2 ./train.sh
GPU_IDS=1 ./train.sh                    # select physical GPU
```

Key env vars: `BACKBONE`, `NUM_EXP`, `BATCH_SIZE`, `LR`, `NC`, `TRAIN_FILE`,
`VAL_FILE`, `TARGET_FOLDER`, `GPU_IDS`, `VIZ_EVERY_ITERS`,
`VIZ_FULL_EVERY_EPOCHS`, `VIZ_NUM_SAMPLES`, `RESUME`.

### YOLO26 alternative

```bash
bash train_yolo.sh                       # YOLO26n on the same source
SCALE=s NUM_EXP=exp2 ./train_yolo.sh     # yolo26s
GPU_IDS=0,1 ./train_yolo.sh              # multi-GPU
```

Reads YOLO labels directly (no conversion needed) and uses Ultralytics
v8DetectionLoss / E2ELoss.

---

## 📊 Visualisation

Per-epoch (and per-N-iters cheap loss refresh), `train.py` writes to
`charts/train/<backbone>/<num_exp>/`:

* `losses.png` — subplots of every tracked loss
* `train_vs_val.png` — mean train loss vs val proxy on a shared axis
* `pr_curve.png`, `recall_f1.png` — detection metrics on the source val set
* `confusion_matrix.png` — normalised, blue colormap
* `samples_day.png` — predictions on well-lit val images
* `samples_synth_night.png` — predictions on Dark ISP-synthesised val images
* `samples_real_night.png` — predictions on real video frames from `target_folder`
* `history.json` — raw loss history for offline plotting

### Standalone Grad-CAM (source vs target feature comparison)

```bash
python -m utils.visualize gradcam \
    --weights weights/dark/dsfd.pth \
    --model dark \
    --num_exp exp1 \
    --target_folder /path/to/target
```

Saves `charts/test/<backbone>/<num_exp>/gradcam_source_vs_target.png`.

---

## 📜 Logging

Every training run mirrors stdout/stderr to a file in `logs/`:

```
logs/<YYYYmmdd_HHMMSS>_<backbone>_<num_exp>.log
```

These contain the argparse config, every loss line, and any Python traceback.

---

## 📑 Citation

This implementation is built on the work of Du et al. If you use it, please
cite both the original paper and this fork:

```bibtex
@inproceedings{du2024boosting,
  title     = {Boosting Object Detection with Zero-Shot Day-Night Domain Adaptation},
  author    = {Du, Zhipeng and Shi, Miaojing and Deng, Jiankang},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages     = {12666--12676},
  year      = {2024}
}
```

This fork is maintained by **Pham Minh Long**, NYCU, Taiwan, as part of
research on low-light object detection on custom datasets (e.g. railway
defect inspection in poor visibility conditions).

---

## 🙏 Acknowledgement

Builds on [DAI-Net](https://github.com/ZPDu/DAI-Net),
[DSFD.pytorch](https://github.com/yxlijun/DSFD.pytorch),
[RetinexNet_PyTorch](https://github.com/aasharma90/RetinexNet_PyTorch),
[MAET](https://github.com/cuiziteng/ICCV_MAET),
[HLA-Face](https://github.com/daooshee/HLA-Face-Code), and
[Ultralytics](https://github.com/ultralytics/ultralytics). Thanks to the
original authors for releasing their code.
