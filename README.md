## DMNIL 2025 [[Paper](https://ieeexplore.ieee.org/document/11540350)] [[arXiv](https://arxiv.org/abs/2502.11381)] [[Models](#pre-trained-checkpoints)] [[Cite](#citation)]

<p align="left">
  <img src="DMNIL/figure/2.png" alt="Xi'an Jiaotong University and AII6 logos" style="width:65%;">
</p>

<h1 align="center">Without Paired Labeled Data: End-to-End Self-Supervised Learning for Drone-View Geo-Localization</h1>

<h3 align="center">
  <strong>Zhongwei Chen</strong><sup>1,2,3</sup>,
  <strong>Zhaoxu Yang*</strong><sup>1,2,3</sup>,
  <strong>Haijun Rong*</strong><sup>1,2,3</sup>,
  <strong>Guoqi Li</strong><sup>4,5,6</sup>
</h3>

<div align="center">
  <sup>1</sup>School of Aerospace Engineering, Xi'an Jiaotong University, China<br>
  <sup>2</sup>State Key Laboratory for Strength and Vibration of Mechanical Structures<br>
  <sup>3</sup>Shaanxi Key Laboratory of Environment and Control for Flight Vehicle<br>
  <sup>4</sup>Institute of Automation, Chinese Academy of Sciences, China<br>
  <sup>5</sup>School of Artificial Intelligence, University of Chinese Academy of Sciences<br>
  <sup>6</sup>Peng Cheng Laboratory<br>
  <sup>*</sup>Corresponding authors
</div>

<div align="center">
  <p>
    <a href="https://ieeexplore.ieee.org/document/11540350"><img src="https://img.shields.io/badge/Paper-IEEE-00629B?logo=ieee&logoColor=white" alt="IEEE paper"></a>
    <a href="https://arxiv.org/abs/2502.11381"><img src="https://img.shields.io/badge/arXiv-2502.11381-B31B1B?logo=arxiv&logoColor=white" alt="arXiv paper"></a>
    <a href="#pre-trained-checkpoints"><img src="https://img.shields.io/badge/Model-Download-2E8B57" alt="Download model"></a>
    <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache%202.0-D22128" alt="Apache 2.0 license"></a>
  </p>
</div>

<p align="center">
  <img src="DMNIL/figure/1_01.png" alt="DMNIL overview, motivation, and University-1652 results" style="width:100%;">
</p>

This repository provides the official PyTorch implementation of **Without Paired Labeled
Data: End-to-End Self-Supervised Learning for Drone-View Geo-Localization**. DMNIL is an
end-to-end self-supervised framework designed to learn cross-view representations without
paired labeled drone and satellite images.

The current release provides the complete University-1652 training and evaluation pipeline,
pre-trained weights, and preprocessing utilities for University-1652, SUES-200, and DenseUAV.

## <a id="news"></a>🔥 News

- **May 21, 2026:** DMNIL was accepted by **IEEE TNNLS 2026**. 🎉
- **September 23, 2025:** The University-1652 model and pre-trained weights were released.
- **September 22, 2025:** Data preprocessing scripts for three DVGL datasets were released.
- **February 17, 2025**: Our  [[arXiv preprint](https://arxiv.org/abs/2502.11381)] was released.
---

## <a id="table-of-contents"></a>📚 Table of Contents

- [Highlights](#highlights)
- [TODOs](#todos)
- [Installation](#installation)
- [Dataset Access](#dataset-access)
- [Dataset Structure](#dataset-structure)
- [Data Preprocessing](#data-preprocessing)
- [Project Structure](#project-structure)
- [Training](#training)
- [Evaluation](#evaluation)
- [Pre-trained Checkpoints](#pre-trained-checkpoints)
- [License](#license)
- [Acknowledgments](#acknowledgments)
- [Citation](#citation)

## <a id="highlights"></a>✨ Highlights

- End-to-end self-supervised drone-view geo-localization without paired labeled data.
- Dynamic Hierarchical Memory Learning (DHML) for cross-view representation learning.
- Information Consistency Evolution Learning (ICEL) for reliable self-supervised optimization.
- A unified University-1652 pipeline for drone-to-satellite and satellite-to-drone retrieval.
- Preprocessing utilities for University-1652, SUES-200, and DenseUAV.
- Released ConvNeXt-Tiny checkpoint for direct evaluation and research reproduction.

## <a id="todos"></a>📜 TODOs

- [x] Release the University-1652 training and evaluation code.
- [x] Release the University-1652 pre-trained checkpoint.
- [x] Release preprocessing scripts for University-1652, SUES-200, and DenseUAV.

## <a id="installation"></a>🛠️ Installation

A CUDA-enabled Linux environment is recommended. Create a Python environment first:

```bash
conda create -n dmnil python=3.9 -y
conda activate dmnil
```

Install the PyTorch build matching your CUDA environment from the
[official PyTorch guide](https://pytorch.org/get-started/locally/), then install the remaining
dependencies:

```bash
pip install timm faiss-gpu scikit-learn scipy albumentations \
  opencv-python imgaug tqdm pillow thop
```

The current release does not include pinned dependency versions. The training entry point
uses multi-GPU `torch.nn.DataParallel` by default.

## <a id="dataset-access"></a>💾 Dataset Access

Please download the datasets from their official project pages:

- [University-1652](https://github.com/layumi/University1652-Baseline)
- [SUES-200](https://github.com/Reza-Zhu/SUES-200-Benchmark)
- [DenseUAV](https://github.com/Dmmm1997/DenseUAV)

Users are responsible for following the licenses and terms of use of the corresponding
datasets.

## <a id="dataset-structure"></a>📁 Dataset Structure

The current training and evaluation pipeline expects University-1652 to follow this layout:

```text
datasets/
└── U1652/
    ├── train/
    │   ├── drone/
    │   │   ├── 0001/
    │   │   ├── 0002/
    │   │   └── ...
    │   ├── satellite_origin/
    │   │   ├── 0001/
    │   │   └── ...
    │   └── satellite/
    │       ├── 0001/
    │       ├── 0002/
    │       └── ...
    └── test/
        ├── query_drone/
        ├── gallery_drone/
        ├── query_satellite/
        └── gallery_satellite/
```

The names of the location subdirectories must be numeric because the current dataset loaders
use the parent directory name as the identity label.

## <a id="data-preprocessing"></a>⚙️ Data Preprocessing

The preprocessing scripts contain source and destination paths near the top of each file.
Update those paths before running the scripts from the repository root.

### University-1652

```bash
python data_process/process_U1652.py
```

### SUES-200

```bash
python data_process/process_SUES-200.py
```

### DenseUAV

```bash
python data_process/process_DenseUAV.py
python data_process/process_DenseUAV_test.py
```

These utilities copy and rename images. Verify their source and destination paths before
execution to avoid writing into an existing processed dataset.

## <a id="project-structure"></a>🗂️ Project Structure

```text
DMNIL-main/
├── DMNIL/
│   ├── dataset/              # Dataset loaders
│   ├── evaluate/             # Retrieval evaluation
│   ├── hand_convnext/        # Modified ConvNeXt backbone
│   ├── solver/               # Optimizers and LR schedulers
│   ├── Utils/                # Sampling, preprocessing, and re-ranking
│   ├── model.py              # Model definitions
│   ├── trainers.py           # DMNIL training objectives
│   └── evaluators.py         # Feature extraction utilities
├── data_process/             # Dataset preprocessing scripts
├── train.py                  # Training and evaluation entry point
├── LICENSE
└── README.md
```

## <a id="training"></a>🚀 Training

### Configure the dataset path

The current University-1652 loaders contain a placeholder root. Replace the following line
in both `DMNIL/dataset/U1652_dor.py` and `DMNIL/dataset/U1652_sat.py`:

```python
root = "/your/path/dataset"  # parent directory containing U1652
```

The entry point uses `--data_dir` for the self-supervised loaders and `--data_folder` for
evaluation data. Set both arguments to the same dataset parent directory.

### University-1652

```bash
python train.py \
  --data_dir /path/to/datasets \
  --data_folder /path/to/datasets
```

The default configuration uses ConvNeXt-Tiny, 384 × 384 input images, 40 epochs, 400
iterations per epoch, and a batch size of 64 per branch. Checkpoints and logs are saved to:

```text
checkpoints/university/convnext-tiny/
```

> **Important:** omit `--only_test` when training. With the current parser,
> `--only_test False` may still be interpreted as `True`.

## <a id="evaluation"></a>📊 Evaluation

### University-1652: Drone to Satellite

```bash
python train.py \
  --only_test True \
  --ckpt_path /path/to/checkpoint.pth \
  --data_dir /path/to/datasets \
  --data_folder /path/to/datasets \
  --dataset U1652-D2S
```

### University-1652: Satellite to Drone

```bash
python train.py \
  --only_test True \
  --ckpt_path /path/to/checkpoint.pth \
  --data_dir /path/to/datasets \
  --data_folder /path/to/datasets \
  --dataset U1652-S2D
```

Run `python train.py --help` to view the complete argument list.

## <a id="pre-trained-checkpoints"></a>🤗 Pre-trained Checkpoints

The University-1652 ConvNeXt-Tiny checkpoint is available from either mirror:

- **Baidu Netdisk:** [DMNIL checkpoint](https://pan.baidu.com/s/13ZKLsXgkQy9Igd7r-ZpUsQ?pwd=6666)
- **Google Drive:** [DMNIL checkpoint](https://drive.google.com/drive/folders/1drUHVCt9hPtPN0b7RmWCT0Wigd6YdJgb?usp=drive_link)

Download the `.pth` file and pass its path to `train.py` through `--ckpt_path`.

## <a id="license"></a>🎫 License

This project is licensed under the [Apache License 2.0](LICENSE).

## <a id="acknowledgments"></a>🙏 Acknowledgments

This repository builds upon ideas and code from
[Sample4Geo](https://github.com/Skyy93/Sample4Geo),
[DAC](https://github.com/SummerpanKing/DAC),
[EM-CVGL](https://github.com/Collebt/EM-CVGL), and
[ADCA](https://github.com/yangbincv/ADCA). We thank the authors for making their excellent
work publicly available.

## <a id="citation"></a>📌 Citation

If you find this work useful in your research, please cite:

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
@ARTICLE{11622851,
  author={Chen, Zhongwei and Yang, Zhao-Xu and Rong, Hai-Jun and Lang, Jiawei and Li, Guoqi},
  journal={IEEE Transactions on Multimedia}, 
  title={From Limited Labels to Open Domains: An Efficient Learning Method for Drone-View Geo-Localization}, 
  year={2026},
  volume={},
  number={},
  pages={1-13},
  keywords={Drones;Educational institutions;Satellites;Learning (artificial intelligence);Location awareness;Modeling;Labeling;Training;Optimization;Visualization;drone-view geo-localization;structural invariance learning;spatial invariance learning;knowledge transfer},
  doi={10.1109/TMM.2026.3716759}}

```
