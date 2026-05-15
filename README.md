<p align="center">
  <h1 align="center">DAI-Net for Low-Light Object Detection — Custom Dataset Adaptation</h1>
  <p align="center">
    <b>Pham Minh Long</b><br>
    National Yang Ming Chiao Tung University (NYCU), Taiwan
  </p>
</p>

PyTorch implementation of low-light object detection with zero-shot day–night
**unsupervised domain adaptation (UDA)**, adapted from the CVPR 2024 paper
**"Boosting Object Detection with Zero-Shot Day-Night Domain Adaptation"**
([arXiv:2312.01220](https://arxiv.org/abs/2312.01220)). This fork extends the
original face-detection pipeline to train on **arbitrary YOLO-format
datasets** (railway defects, custom objects, …) with a **real low-light
target domain** sampled from video, and adds explicit UDA objectives.

![overview](./assets/overview.png)

---

## ✨ Highlights

| Area | Original (Du et al.) | This fork |
|---|---|---|
| Detection task | Face only | Any YOLO-format multi-class dataset |
| Source domain | WIDER Face | User well-lit + labelled images |
| Target domain | Synthetic Dark-ISP only | **Real night frames + synthetic Dark-ISP** |
| Detector | DSFD only | DSFD / DAI-Net **or** YOLO26 (one entrypoint) |
| Domain adaptation | Synthetic only | + cross-domain **KL**, **entropy-min**, **weight anchor**, target Retinex |
| Metrics | — | scikit-learn precision/recall/F1, PR/AP |
| Visualisation | — | losses, PR/CM/F1, t-SNE, Grad-CAM, domain-metric curves |
| Layout | flat scripts | clean **`src/` package layout**, YAML-driven |

---

## 📁 Project layout

```
DAI-Net/
├── train.py                 # unified entrypoint (dispatches by architecture)
├── test.py                  # evaluation entrypoint
├── configs/                 # YAML experiment configs
├── dataset/                 # source_train.txt / source_val.txt (abs paths)
├── weights/                 # pretrained + checkpoints  <arch>/<backbone>/<exp>/
├── charts/  records/  logs/ # per-experiment outputs
└── src/
    ├── data/                # ALL dataset loading
    │   ├── source_domain.py     labelled source (BGR-CHW)
    │   ├── target_domain.py     TargetDomainDetection + TargetUnlabeledDataset
    │   ├── datamodule.py        DataModule (Facade: builds every loader)
    │   ├── meta.py              data.yaml -> nc / class names
    │   └── config.py            global cfg (INPUT_SIZE, loss weights, schedule)
    ├── utils/               # augmentations, dark_isp, visualize, convert/cut tools
    ├── scripts/             # *.sh launchers
    └── models/
        ├── dai_net.py  dsfd_*.py  enhancer.py  factory.py
        ├── layers/          detection layers / losses (multibox, enhance, …)
        ├── yolo/            vendored Ultralytics (YOLO26)
        └── dainet/          framework package
            ├── config/          Config dataclass + ConfigLoader (Builder)
            ├── constants.py     name/arch/backbone maps, CSV columns
            └── yolo_runner.py   YOLO training (config-driven)
```

`train.py` / `test.py` prepend `src/` and `src/models/` to `sys.path`, so every
absolute import (`from data.config import cfg`, `from models.factory import …`)
resolves with no per-file path hacks.

---

## ⚙️ Installation

```bash
conda create -n dainet python=3.10 -y && conda activate dainet
pip install -r requirements.txt          # torch, torchvision, sklearn, …
```

Place pretrained weights in `weights/`:
`vgg16_reducedfc.pth` (backbone) and `decomp.pth` (frozen Retinex DecomNet).

---

## 🗂️ Data

* **Source** (labelled, well-lit): listed in
  `dataset/source_train.txt` / `source_val.txt`, one line per image:

  ```
  /abs/path/img.png  N  x y w h cls  x y w h cls  ...
  ```

  Class ids are 1-indexed (0 = background). Convert YOLO labels with
  `src/utils/convert_yolo_to_dainet.py`. Paths must contain **no spaces**
  (symlink a space-free path if your drive label has spaces).
* **Target** (unlabeled, real night): a flat folder of frames
  (`src/utils/cut_frames.py` extracts them from video).
* `data.yaml` (in `source_folder`) supplies `nc` + `names`.

---

## 🚀 Training

One YAML drives everything; `architecture:` selects the pipeline.

```bash
# DAI-Net / DSFD
python train.py --config configs/train/dai_net/vgg16/exp1.yaml
# YOLO26  (architecture: yolo… in the YAML -> YOLO runner)
python train.py --config configs/train/yolo/csp/exp1.yaml
```

Multi-GPU: `torchrun --nproc_per_node=N train.py --config …`.

### Loss composition (DAI-Net path)

Per iteration the model sees **three domains** — source (labelled day),
`source_dark` (Dark-ISP synthetic, paired GT), and real `target` (unlabeled):

| Term | Purpose |
|---|---|
| `pal1/pal2 loc+conf` | detection on `source_dark` vs source GT (paired) |
| `enhance`, `enhance_l1ssim` | Retinex reconstruction (paper-faithful) |
| `mutual` | KL: source ↔ source_dark backbone features |
| `kl_st` | **UDA** KL: source ↔ target features (`kl_loss_weight`) |
| `target_unsup` | Retinex reconstruction on **real target** via the *trainable* reflectance branch (`target_loss_weight`) |
| `wreg` | 3C-GAN ℓ_wReg — anchor VGG to pretrained (`wreg_loss_weight`) |
| `entropy` | 3C-GAN ℓ_ent — entropy-min on target detection (`entropy_loss_weight`) |

Set any weight to `0` in the config to disable that term.

---

## 📊 Evaluation & visualisation

```bash
python test.py --config configs/test/dai_net/vgg16/exp1.yaml
```

Charts land in `charts/<mode>/<arch>/<backbone>/<exp>/`:

* `losses.png`, `train_vs_val.png` (pal2-det vs val, like-for-like)
* `confusion_matrix.png`, `pr_curve.png`, `recall_f1.png`
  (precision/recall/F1 & AP via **scikit-learn**)
* `samples_{day,synth_night,real_night}.png`
* `tsne_source_features.png` — source embeddings over the **full** val set,
  coloured by GT class
* `gradcam_source_vs_target.png`
* `domain_metrics.png` — KL & target-entropy vs epoch

Validation prints a sectioned box table (Loss / Detection / Domain / Timing)
and every visualisation step logs progress to stdout.

CSV metrics: `records/<arch>/<backbone>/<exp>_{train,val}.csv`.

---

## 🔧 Config knobs (excerpt)

```yaml
architecture: dai_net           # or "yolo" -> YOLO26 runner
num_exp: exp1
batch_size: 2
lr: 5.0e-4
epochs: 100
lr_steps: [170000, 250000, 310000]
source_folder: /path/no-spaces/.../3/source     # holds Train/ Val/ data.yaml
target_folder: /path/no-spaces/.../3/target     # real night frames
kl_loss_weight: 1.0
target_loss_weight: 0.05
wreg_loss_weight: 1.0e-4
entropy_loss_weight: 0.01
resume: false                   # true -> auto-load last checkpoint
```

Checkpoints: `weights/<arch>/<backbone>/<num_exp>/{last,best}_model.pth`.

---

## 🙏 Acknowledgements

Built on DAI-Net (Du et al., CVPR 2024), DSFD, and Ultralytics YOLO. The
unsupervised model-adaptation regularisers follow *"Model Adaptation:
Unsupervised Domain Adaptation without Source Data"* (Li et al., CVPR 2020).
