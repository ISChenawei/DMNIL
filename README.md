<div align="center">
  <img src="DMNIL/figure/2.png" alt="Xi'an Jiaotong University and AII6 logos" width="620">

  <h1>DMNIL</h1>
  <h3>Without Paired Labeled Data: End-to-End Self-Supervised Learning<br>for Drone-View Geo-Localization</h3>

  <p>
    <strong>Zhongwei Chen</strong><sup>1,2,3</sup> ·
    <strong>Zhaoxu Yang*</strong><sup>1,2,3</sup> ·
    <strong>Haijun Rong*</strong><sup>1,2,3</sup> ·
    <strong>Guoqi Li</strong><sup>4,5,6</sup>
  </p>

  <details>
    <summary>Affiliations</summary>
    <sub>
      <sup>1</sup>School of Aerospace Engineering, Xi'an Jiaotong University, China<br>
      <sup>2</sup>State Key Laboratory for Strength and Vibration of Mechanical Structures<br>
      <sup>3</sup>Shaanxi Key Laboratory of Environment and Control for Flight Vehicle<br>
      <sup>4</sup>Institute of Automation, Chinese Academy of Sciences, China<br>
      <sup>5</sup>School of Artificial Intelligence, University of Chinese Academy of Sciences<br>
      <sup>6</sup>Peng Cheng Laboratory
    </sub>
  </details>

  <p>
    <a href="https://ieeexplore.ieee.org/document/11540350"><img src="https://img.shields.io/badge/Paper-IEEE-00629B?logo=ieee&logoColor=white" alt="IEEE paper"></a>
    <a href="https://arxiv.org/abs/2502.11381"><img src="https://img.shields.io/badge/arXiv-2502.11381-B31B1B?logo=arxiv&logoColor=white" alt="arXiv paper"></a>
    <a href="#pre-trained-model"><img src="https://img.shields.io/badge/Model-Download-2E8B57" alt="Download model"></a>
    <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache%202.0-D22128" alt="Apache 2.0 license"></a>
  </p>

  <p>
    <a href="#quick-start">Quick Start</a> ·
    <a href="#data-preparation">Data</a> ·
    <a href="#training">Training</a> ·
    <a href="#evaluation">Evaluation</a> ·
    <a href="#citation">Citation</a>
  </p>
</div>

---

## Overview

This repository is the official implementation of **DMNIL**, an end-to-end self-supervised method for drone-view geo-localization without paired labeled data.

The released training and evaluation pipeline currently targets **University-1652**. Data preprocessing tools for **University-1652**, **SUES-200**, and **DenseUAV** are also included.

<p align="center">
  <img src="DMNIL/figure/1_01.png" alt="Overview and performance of the proposed DMNIL method" width="100%">
</p>

## News

- **2026-05-21** — DMNIL was accepted by **IEEE TNNLS 2026**. 🎉
- **2025-09-23** — University-1652 pre-trained weights were released.
- **2025-09-22** — Data preprocessing scripts were released.

## Quick start

### 1. Environment

```bash
conda create -n dmnil python=3.9 -y
conda activate dmnil
```

Install a CUDA-compatible build of [PyTorch](https://pytorch.org/get-started/locally/), followed by the remaining dependencies:

```bash
pip install timm faiss-gpu scikit-learn scipy albumentations \
  opencv-python imgaug tqdm pillow thop
```

The current code is intended for a CUDA-enabled Linux environment and multi-GPU `DataParallel` training. Dependency versions are not pinned in this release.

### 2. Dataset

Download University-1652 and arrange it under a common dataset root:

```text
/path/to/datasets/U1652/
├── train/
│   ├── drone/
│   └── satellite/
└── test/
    ├── query_drone/
    ├── gallery_satellite/
    ├── query_satellite/
    └── gallery_drone/
```

<details>
<summary><strong>Complete directory layout</strong></summary>

```text
/path/to/datasets/
└── U1652/
    ├── train/
    │   ├── drone/
    │   │   ├── 0001/
    │   │   └── ...
    │   ├── satellite_origin/
    │   │   ├── 0001/
    │   │   └── ...
    │   └── satellite/
    │       ├── 0001/
    │       └── ...
    └── test/
        ├── query_drone/
        ├── gallery_satellite/
        ├── query_satellite/
        └── gallery_drone/
```

</details>

### 3. Paths

Before running the current release, replace the placeholder below in both `DMNIL/dataset/U1652_dor.py` and `DMNIL/dataset/U1652_sat.py`:

```python
root = "/your/path/dataset"  # parent directory of U1652
```

Use the same parent directory for `--data_dir` and `--data_folder` in the commands below.

## Data preparation

Download each dataset from its official project page, update the source and destination paths inside the corresponding script, and then run it from the repository root.

| Dataset | Official project | Preprocessing command |
| :--- | :--- | :--- |
| University-1652 | [Dataset](https://github.com/layumi/University1652-Baseline) | `python data_process/process_U1652.py` |
| SUES-200 | [Dataset](https://github.com/Reza-Zhu/SUES-200-Benchmark) | `python data_process/process_SUES-200.py` |
| DenseUAV | [Dataset](https://github.com/Dmmm1997/DenseUAV) | `python data_process/process_DenseUAV.py` |

DenseUAV also provides a test-set organization script:

```bash
python data_process/process_DenseUAV_test.py
```

The preprocessing scripts copy and rename images. Review their source and destination paths before execution to avoid writing into an existing processed dataset.

## Training

Run training from the repository root:

```bash
python train.py \
  --data_dir /path/to/datasets \
  --data_folder /path/to/datasets
```

The default configuration uses ConvNeXt-Tiny, 384 × 384 images, 40 epochs, 400 iterations per epoch, and a batch size of 64 per branch. Outputs are saved under:

```text
checkpoints/university/convnext-tiny/
```

> To train, omit `--only_test`. With the current argument parser, `--only_test False` may still be interpreted as `True`.

## Evaluation

### Drone → Satellite

```bash
python train.py \
  --only_test True \
  --ckpt_path /path/to/checkpoint.pth \
  --data_dir /path/to/datasets \
  --data_folder /path/to/datasets \
  --dataset U1652-D2S
```

### Satellite → Drone

```bash
python train.py \
  --only_test True \
  --ckpt_path /path/to/checkpoint.pth \
  --data_dir /path/to/datasets \
  --data_folder /path/to/datasets \
  --dataset U1652-S2D
```

Run `python train.py --help` to view all available options.

## Pre-trained model

| Dataset | Backbone | Baidu Netdisk | Google Drive |
| :--- | :---: | :---: | :---: |
| University-1652 | ConvNeXt-Tiny | [Download](https://pan.baidu.com/s/13ZKLsXgkQy9Igd7r-ZpUsQ?pwd=6666) | [Download](https://drive.google.com/drive/folders/1drUHVCt9hPtPN0b7RmWCT0Wigd6YdJgb?usp=drive_link) |

Pass the downloaded `.pth` file to `--ckpt_path` when evaluating.

## Release status

| Component | University-1652 | SUES-200 | DenseUAV |
| :--- | :---: | :---: | :---: |
| Preprocessing | ✅ | ✅ | ✅ |
| Dataset loader | ✅ | ✅ | ✅ |
| Training configuration | ✅ | Planned | Planned |
| Pre-trained model | ✅ | — | — |

<details>
<summary><strong>Current implementation notes</strong></summary>

- Dataset and preprocessing paths require manual configuration.
- Dependency versions are not pinned yet.
- The current model setup assumes multi-GPU `DataParallel` execution; single-GPU use may require a small code adjustment.
- The released `train.py` pipeline is configured for University-1652. Complete SUES-200 and DenseUAV training recipes will be added in a future update.

</details>

## Citation

If you find this project useful, please cite our work:

```bibtex
@ARTICLE{11540350,
  author={Chen, Zhongwei and Yang, Zhao-Xu and Rong, Hai-Jun and Li, Guoqi},
  journal={IEEE Transactions on Neural Networks and Learning Systems},
  title={Without Paired Labeled Data: End-to-End Self-Supervised Learning for Drone-View Geo-Localization},
  year={2026},
  volume={},
  number={},
  pages={1-15},
  keywords={Drones;Learning (artificial intelligence);Satellites;Labeling;Self-supervised learning;Modeling;Educational institutions;Training;Location awareness;Memory;Drone-view geo-localization (DVGL);dynamic hierarchical memory learning (DHML);information consistency evolution learning (ICEL);self-supervised learning},
  doi={10.1109/TNNLS.2026.3696684}
}
```

This repository also builds on our previous work, **CDIKTNet**:

```bibtex
@article{chen2025limited,
  title={From limited labels to open domains: An efficient learning method for drone-view geo-localization},
  author={Chen, Zhongwei and Yang, Zhao-Xu and Rong, Hai-Jun and Lang, Jiawei and Li, Guoqi},
  journal={arXiv preprint arXiv:2503.07520},
  year={2025}
}
```

## Acknowledgments

This repository builds on [Sample4Geo](https://github.com/Skyy93/Sample4Geo), [DAC](https://github.com/SummerpanKing/DAC), [EM-CVGL](https://github.com/Collebt/EM-CVGL), and [ADCA](https://github.com/yangbincv/ADCA). We thank the authors for their excellent work.

## License

Released under the [Apache License 2.0](LICENSE).
