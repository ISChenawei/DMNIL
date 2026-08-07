# DMNIL

<div align="center">
  <h2>Without Paired Labeled Data: End-to-End Self-Supervised Learning for Drone-View Geo-Localization</h2>

  <p>
    <a href="https://ieeexplore.ieee.org/document/11540350">IEEE paper</a> |
    <a href="https://arxiv.org/abs/2502.11381">arXiv</a> |
    <a href="#pre-trained-checkpoints">Pre-trained models</a> |
    <a href="#citation">Citation</a>
  </p>
</div>

<p align="center">
  <img src="DMNIL/figure/2.png" alt="Overview of the DMNIL method" width="50%">
</p>

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
  <sup>6</sup>Peng Cheng Laboratory
</div>

<p align="center">
  <img src="DMNIL/figure/1_01.png" alt="DMNIL framework" width="100%">
</p>

This repository is the official implementation of **DMNIL**, the end-to-end self-supervised method introduced in the paper *Without Paired Labeled Data: End-to-End Self-Supervised Learning for Drone-View Geo-Localization*.

The released training and evaluation pipeline currently targets **University-1652**. Data preprocessing utilities for University-1652, SUES-200, and DenseUAV are included; complete training configurations for SUES-200 and DenseUAV are not yet provided.

## Table of contents

- [News](#news)
- [Installation](#installation)
- [Dataset access](#dataset-access)
- [Dataset preparation](#dataset-preparation)
- [Training and evaluation](#training-and-evaluation)
- [Pre-trained checkpoints](#pre-trained-checkpoints)
- [Current limitations](#current-limitations)
- [License](#license)
- [Citation](#citation)
- [Acknowledgments](#acknowledgments)

## News

- **May 21, 2026:** DMNIL was accepted by IEEE TNNLS 2026. 🎉
- **September 23, 2025:** We released the DMNIL model trained on University-1652 and its pre-trained weights.
- **September 22, 2025:** We released data preprocessing scripts for University-1652, SUES-200, and DenseUAV.

## Installation

Clone or download this repository, then create a Python environment:

```bash
conda create -n dmnil python=3.9 -y
conda activate dmnil
```

Install a CUDA-compatible version of PyTorch and torchvision by following the [official PyTorch installation guide](https://pytorch.org/get-started/locally/). Then install the remaining dependencies used by the repository:

```bash
pip install timm faiss-gpu scikit-learn scipy albumentations opencv-python imgaug tqdm pillow thop
```

> [!NOTE]
> The repository does not currently provide pinned dependency versions. A CUDA-enabled Linux environment is recommended. The current training entry point is designed around multi-GPU `torch.nn.DataParallel` execution.

## Dataset access

Please download the datasets from their official project pages:

- [University-1652](https://github.com/layumi/University1652-Baseline)
- [SUES-200](https://github.com/Reza-Zhu/SUES-200-Benchmark)
- [DenseUAV](https://github.com/Dmmm1997/DenseUAV)

Users are responsible for following the licenses and terms of use of the corresponding datasets.

## Dataset preparation

### University-1652

The University-1652 directory used by the current training and evaluation code should follow this structure:

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

The preprocessing scripts currently use paths defined inside each script. Update `original_dataset_root` and `target_dataset_root` in the script before running it:

```bash
python data_process/process_U1652.py
```

### SUES-200

Update the source and target paths in `data_process/process_SUES-200.py`, then run:

```bash
python data_process/process_SUES-200.py
```

### DenseUAV

Update the source and target paths in the corresponding scripts, then run:

```bash
python data_process/process_DenseUAV.py
python data_process/process_DenseUAV_test.py
```

> [!IMPORTANT]
> The preprocessing utilities copy and rename images into the layout expected by the loaders. To avoid overwriting an existing processed dataset, review the source and destination paths before executing them.

## Training and evaluation

### Configure University-1652 paths

The current University-1652 training loaders contain a placeholder root path. Before running the code, replace the following line in both `DMNIL/dataset/U1652_dor.py` and `DMNIL/dataset/U1652_sat.py`:

```python
root = "/your/path/dataset"
```

The value must be the parent directory containing the `U1652` folder, for example `/data/datasets` when the dataset is stored at `/data/datasets/U1652`.

The entry point also uses two command-line paths:

- `--data_dir`: parent directory used by the self-supervised training loaders.
- `--data_folder`: parent directory used to construct the evaluation paths.

For the current implementation, set both arguments to the same dataset parent directory.

### Train

From the repository root, run:

```bash
python train.py \
  --data_dir /path/to/datasets \
  --data_folder /path/to/datasets
```

Checkpoints and logs are written to:

```text
checkpoints/university/convnext-tiny/
```

The main defaults are 40 epochs, 400 iterations per epoch, an input resolution of 384 × 384, and a batch size of 64 per branch. Run `python train.py --help` for the complete argument list.

> [!IMPORTANT]
> To start training, omit `--only_test`. Do not use `--only_test False`: the current argument parser uses `type=bool`, and a non-empty string such as `False` may still be interpreted as `True`.

### Evaluate a checkpoint

Drone-to-satellite retrieval is the default evaluation direction:

```bash
python train.py \
  --only_test True \
  --ckpt_path /path/to/checkpoint.pth \
  --data_dir /path/to/datasets \
  --data_folder /path/to/datasets \
  --dataset U1652-D2S
```

For satellite-to-drone retrieval, use:

```bash
python train.py \
  --only_test True \
  --ckpt_path /path/to/checkpoint.pth \
  --data_dir /path/to/datasets \
  --data_folder /path/to/datasets \
  --dataset U1652-S2D
```

The evaluator reports Recall@1, Recall@5, Recall@10, and related retrieval metrics in the console and log file.

## Pre-trained checkpoints

Pre-trained University-1652 checkpoints are available from either mirror:

| Dataset | Backbone | Download |
| --- | --- | --- |
| University-1652 | ConvNeXt-Tiny | [Baidu Netdisk](https://pan.baidu.com/s/13ZKLsXgkQy9Igd7r-ZpUsQ?pwd=6666) · [Google Drive](https://drive.google.com/drive/folders/1drUHVCt9hPtPN0b7RmWCT0Wigd6YdJgb?usp=drive_link) |

Use the downloaded `.pth` file with the `--ckpt_path` argument shown above.

## Current limitations

- The released `train.py` pipeline is currently configured for University-1652.
- Complete SUES-200 and DenseUAV training commands and configurations are not yet included.
- Dataset and preprocessing paths still need to be configured manually.
- Dependency versions are not pinned yet.
- The current model setup assumes multi-GPU `DataParallel` execution; single-GPU support may require a small code adjustment.

We will continue improving the repository for clarity and reproducibility.

## License

This project is licensed under the [Apache License 2.0](LICENSE).

## Citation

If you find this code useful for your research, please cite the DMNIL paper:

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

This code is also based on our previous work, CDIKTNet:

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
