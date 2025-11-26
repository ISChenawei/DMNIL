## DMNIL 2025 [[paper](https://arxiv.org/abs/2502.11381)][[model](#pre-trained-checkpoints)] [[Cite](#Citation)]
<p align="center">
  <p align="left">
    <img src="DMNIL/figure/2.png" alt="Description of the image" style="width:50%;">
  <p align="left">
<h1 align="center">End-to-End Self-Supervised Method for Drone-View Geo-Localization</h1>
<h3 align="center"><strong>Zhongwei Chen</strong><sup>1,2,3</sup>, <strong>Zhaoxu Yang*</strong><sup>1,2,3</sup>, <strong>Haijun Rong*</strong><sup>1,2,3</sup>, <strong>Guoqi Li</strong><sup>4,5,6</sup></h3>

<div align="center">
  <sup>1</sup>School of Aerospace Engineering, Xi'an Jiaotong University China<br>
  <sup>2</sup>State Key Laboratory for Strength and Vibration of Mechanical Structures<br>
  <sup>3</sup>Shaanxi Key Laboratory of Environment and Control for Flight Vehicle<br>
  <sup>4</sup>Institute of Automation, Chinese Academy of Sciences, China<br>
  <sup>5</sup>School of Artificial Intelligence, University of Chinese Academy of Sciences<br>
  <sup>6</sup>Peng Cheng Laboratory
</div>
  <p align="center">
    <img src="DMNIL/figure/1_01.png" alt="Description of the image" style="width:100%;">
  <p align="center">

This repository is the official implementation of the paper "Without Paired Labeled Data: End-to-End Self-Supervised Method for Drone-View Geo-Localization" (https://arxiv.org/abs/2502.11381). 
The current version of the repository can cover the experiments reported in the paper, for researchers in time efficiency. And we will also update this repository for better understanding and clarity.

## <a id="table-of-contents"></a> 📚 Table of contents

- [Dataset Access](#dataset-access)
- [Dataset Structure](#dataset-structure)
- [Train and Test](#train-and-test)
- [Pre-trained Checkpoints](#pre-trained-checkpoints)
- [License](#license)
- [Acknowledgments](#acknowledgments)
- [Citation](#citation)

## <a id="dataset-access"></a> 💾 Dataset Access
Please prepare [University-1652](https://github.com/layumi/University1652-Baseline), [SUES-200](https://github.com/Reza-Zhu/SUES-200-Benchmark), [DenseUAV](https://github.com/Dmmm1997/DenseUAV)
## <a id="dataset-structure"></a> 📁 Dataset Process

### Process University-1652 Dataset  
```
run process_data/porcess_U1652.py
```
### Process SUES-200 Dataset  
```
run process_data/porcess_SUES-200.py
```
### Process DenseUAV Dataset 
```
run process_data/porcess_SUES-200.py
```
## <a id="train-and-test"></a> 🚀 Train and Test

#### University-1652 Dataset Configuration
```
1. Set Dataset Paths
# 1. U1652_dro.py — Drone-view data loader
root = '/your/path/dataset'

# 2. U1652_sat.py — Satellite-view data loader
root = '/your/path/dataset'

# 3. train.py — Global dataset directory (used for testing)
parser.add_argument(
    '--data_dir',
    default='/your/path/dataset',
    type=str
)

2. Training and Testing

python train.py --only_test False

python train.py --only_test True --ckpt_path /path/to/your/checkpoint.pth
```

#### SUES-200 Dataset Configuration
```
The remaining code is scheduled to be updated in the near future.
```
#### DenseUAV Dataset Configuration
```
The remaining code is scheduled to be updated in the near future.
```
## <a id="pre-trained-checkpoints"></a> 🤗 Pre-trained Checkpoints
We provide the trained models in the link below:

https://pan.baidu.com/s/151fhP4kwW4rTXqjRRlSM6Q 提取码: sm8s 

We will update this repository for better clarity ASAP, current version is for quick research for researchers interested in the cross-view geo-localization task.

## <a id="license"></a> 🎫 License
This project is licensed under the [Apache 2.0 license](LICENSE).

## <a id="citation"></a> 📌 Citation

 If you find this code useful for your research, please cite our papers.

```bibtex
@article{chen2025without,
  title={Without Paired Labeled Data: End-to-End Self-Supervised Method for Drone-View Geo-Localization},
  author={Chen, Zhongwei, Yang, Zhao-Xu, Rong, Hai-Jun, Guoqi Li},
  journal={arXiv preprint arXiv:2502.11381},
  year={2025}
}
```
This code is based on previous work CDIKTNet. If you find this code useful for your research, please cite our papers.

```bibtex
@Article{chen2025limited,
title={From Limited Labels to Open Domains: An Efficient Learning Method for Drone-view Geo-Localization},
author={Chen, Zhongwei, Yang, Zhao-Xu, Rong, Hai-Jun, Lang, Jiawei, Li, Guoqi},
journal={arXiv preprint arXiv:2503.07520},
year={2025}
}
```
## <a id="acknowledgments"></a> 🙏 Acknowledgments
This repository is built using the Sample4Geo[https://github.com/Skyy93/Sample4Geo], DAC [https://github.com/SummerpanKing/DAC], EM-CVGL [https://github.com/Collebt/EM-CVGL], ADCA [https://github.com/yangbincv/ADCA] repositories.
