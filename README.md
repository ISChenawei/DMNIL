## DMNIL 2025 [[paper](https://arxiv.org/abs/2502.11381)][[model](#pre-trained-checkpoints)] [[Cite](#Citation)]
This repository is the official implementation of the paper "Without Paired Labeled Data: An End-to-End Self-Supervised Paradigm for UAV-View Geo-Localization" (https://arxiv.org/abs/2502.11381). 

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
## <a id="dataset-structure"></a> 📁 Dataset Structure

### University-1652 Dataset Directory Structure
```
├── University-1652/
│   ├── train/
│       ├── drone/                   /* drone-view training images 
│           ├── 0001
|           ├── 0002
|           ...
│       ├── satellite/               /* satellite-view training images       
│   ├── test/
│       ├── query_drone/  
│       ├── gallery_drone/  
│       ├── query_satellite/  
│       ├── gallery_satellite/ 
```
### SUES-200 Dataset Directory Structure
```
├─ SUES-200
  ├── Training
    ├── 150/
    ├── 200/
    ├── 250/
    └── 300/
  ├── Testing
    ├── 150/
    ├── 200/ 
    ├── 250/	
    └── 300/
```
### DenseUAV Dataset Directory Structure
```
├── DenseUAV/
│   ├── Dense_GPS_ALL.txt           /* format as: path latitude longitude height
│   ├── Dense_GPS_test.txt
│   ├── Dense_GPS_train.txt
│   ├── train/
│       ├── drone/                   /* drone-view training images
│           ├── 000001
│               ├── H100.JPG
│               ├── H90.JPG
│               ├── H80.JPG
|           ...
│       ├── satellite/               /* satellite-view training images
│           ├── 000001
│               ├── H100_old.tif
│               ├── H90_old.tif
│               ├── H80_old.tif
│               ├── H100.tif
│               ├── H90.tif
│               ├── H80.tif
|           ...
│   ├── test/
│       ├── query_drone/             /* UAV-view testing images
│       ├── query_satellite/         /* satellite-view testing images
```
## <a id="train-and-test"></a> 🚀 Train and Test

For University-1652 Dataset
```
Train: run train_university.py, with --only_test = False.

Test: run train_university.py, with --only_test = True, and choose the model in --ckpt_path.
```
For SUES-200 Dataset
```
Train: run train_SUES-200.py, with --only_test = False.

Test: run train_SUES-200.py, with --only_test = True, and choose the model in --ckpt_path.
```
For DenseUAV Dataset
```
Train: run train_DenseUAV.py, with --only_test = False.

Test: run train_DenseUAV.py, with --only_test = True, and choose the model in --ckpt_path.
```
## <a id="pre-trained-checkpoints"></a> 🤗 Pre-trained Checkpoints
We provide the trained models in the link below:

Link: The trained models will be released soon.

We will update this repository for better clarity ASAP, current version is for quick research for researchers interested in the cross-view geo-localization task.

## <a id="license"></a> 🎫 License
This project is licensed under the [Apache 2.0 license](LICENSE).

## <a id="citation"></a> 📌 Citation

 If you find this code useful for your research, please cite our papers.

```bibtex
@article{chen2025without,
  title={Without Paired Labeled Data: An End-to-End Self-Supervised Paradigm for UAV-View Geo-Localization},
  author={Chen, Zhongwei and Yang, Zhao-Xu and Rong, Hai-Jun},
  journal={arXiv preprint arXiv:2502.11381},
  year={2025}
}
```

## <a id="acknowledgments"></a> 🙏 Acknowledgments
This repository is built using the Sample4Geo[https://github.com/Skyy93/Sample4Geo], DAC [https://github.com/SummerpanKing/DAC], EM-CVGL [https://github.com/Collebt/EM-CVGL], ADCA [https://github.com/yangbincv/ADCA]repositories.
