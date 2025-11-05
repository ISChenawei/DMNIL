import os
import time
import shutil
import sys
from datetime import timedelta
import numpy as np
import torch
import argparse
import os.path as osp
from bisect import bisect_right

from sklearn.cluster import DBSCAN
from torch import nn
from dataclasses import dataclass
import collections
from torch.utils.data import DataLoader
import torch.nn.functional as F
from DMNIL.evaluators import extract_features
from DMNIL.hand_convnext.ConvNext.cluster_model import ClusterMemory, ClusterMemoryV2, DHML
from DMNIL.solver import make_optimizer, WarmupMultiStepLR
from DMNIL.Utils.init import IterLoader
from DMNIL.Utils.sampler import RandomMultipleGallerySamplerNoCam, RandomMultipleGallerySampler
from DMNIL.Utils.preprocessor import Preprocessor,Preprocessor_color
from DMNIL.dataset.university import U1652DatasetEval, get_transforms
from DMNIL import dataset
from DMNIL.traners import Trainer_DMNIL
from DMNIL.utils import setup_system, Logger
import re
from DMNIL.evaluate.university import evaluate
from DMNIL.model import TimmModel
from DMNIL.Utils import transforms as T
from DMNIL.Utils.faiss_rerank import compute_jaccard_distance

def get_data(name, data_dir):
    root = osp.join(data_dir, name)
    datasets = dataset.create(name, root)
    return datasets

def get_train_loader_dro(args, dataset, height, width, batch_size, workers,
                     num_instances, iters, trainset=None, no_cam=False,train_transformer=None):

    train_set = sorted(dataset.train) if trainset is None else sorted(trainset)
    rmgs_flag = num_instances > 0
    if rmgs_flag:
        if no_cam:
            sampler = RandomMultipleGallerySamplerNoCam(train_set, num_instances)
        else:
            sampler = RandomMultipleGallerySampler(train_set, num_instances)
    else:
        sampler = None
    train_loader = IterLoader(
        DataLoader(Preprocessor(train_set, root=dataset.images_dir, transform=train_transformer),
                   batch_size=batch_size, num_workers=workers, sampler=sampler,
                   shuffle=not rmgs_flag, pin_memory=True, drop_last=True), length=iters)

    return train_loader

def get_train_loader_sat(args, dataset, height, width, batch_size, workers,
                     num_instances, iters, trainset=None, no_cam=False,train_transformer=None,train_transformer1=None):

    train_set = sorted(dataset.train) if trainset is None else sorted(trainset)
    rmgs_flag = num_instances > 0
    if rmgs_flag:
        if no_cam:
            sampler = RandomMultipleGallerySamplerNoCam(train_set, num_instances)
        else:
            sampler = RandomMultipleGallerySampler(train_set, num_instances)
    else:
        sampler = None
    if train_transformer1 is None:
        train_loader = IterLoader(
            DataLoader(Preprocessor(train_set, root=dataset.images_dir, transform=train_transformer),
                       batch_size=batch_size, num_workers=workers, sampler=sampler,
                       shuffle=not rmgs_flag, pin_memory=True, drop_last=True), length=iters)
    else:
        train_loader = IterLoader(
            DataLoader(Preprocessor_color(train_set, root=dataset.images_dir, transform=train_transformer,transform1=train_transformer1),
                       batch_size=batch_size, num_workers=workers, sampler=sampler,
                       shuffle=not rmgs_flag, pin_memory=True, drop_last=True), length=iters)

    return train_loader


def get_test_loader(dataset, height, width, batch_size, workers, testset=None,test_transformer=None):

    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    if test_transformer is None:
        test_transformer = T.Compose([
            T.Resize((height, width), interpolation=3),
            T.ToTensor(),
            normalizer
        ])
    if testset is None:
        testset = list(set(dataset.query) | set(dataset.gallery))

    test_loader = DataLoader(
        Preprocessor(testset, root=dataset.images_dir, transform=test_transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=True)

    return test_loader


def load_latest_checkpoint(config):
    ckpt_dir = os.path.join(config.model_path,config.model)

    files = os.listdir(ckpt_dir)

    weight_files = [f for f in files if f.startswith("weights_e") and f.endswith(".pth")]

    if not weight_files:
        raise FileNotFoundError(f"No checkpoint found in {ckpt_dir}")

    epoch_numbers = []
    for f in weight_files:
        match = re.search(r"weights_e(\d+)\.pth", f)
        if match:
            epoch_numbers.append((int(match.group(1)), f))

    if not epoch_numbers:
        raise FileNotFoundError("No valid checkpoint file found!")

    latest_epoch, latest_file = max(epoch_numbers, key=lambda x: x[0])
    checkpoint_path = os.path.join(ckpt_dir, latest_file)

    print(f"Loading latest checkpoint: {checkpoint_path}")


    return checkpoint_path, latest_epoch


class WarmupMultiStepLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(
        self,
        optimizer,
        milestones,
        gamma=0.1,
        warmup_factor=1.0 / 3,
        warmup_iters=500,
        warmup_method="linear",
        last_epoch=-1,
    ):
        if not list(milestones) == sorted(milestones):
            raise ValueError(
                "Milestones should be a list of" " increasing integers. Got {}",
                milestones,
            )

        if warmup_method not in ("constant", "linear"):
            raise ValueError(
                "Only 'constant' or 'linear' warmup_method accepted"
                "got {}".format(warmup_method)
            )
        self.milestones = milestones
        self.gamma = gamma
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        self.warmup_method = warmup_method
        super(WarmupMultiStepLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        warmup_factor = 1
        if self.last_epoch < self.warmup_iters:
            if self.warmup_method == "constant":
                warmup_factor = self.warmup_factor
            elif self.warmup_method == "linear":
                alpha = float(self.last_epoch) / float(self.warmup_iters)
                warmup_factor = self.warmup_factor * (1 - alpha) + alpha
        return [
            base_lr
            * warmup_factor
            * self.gamma ** bisect_right(self.milestones, self.last_epoch)
            for base_lr in self.base_lrs
        ]


class Normalize(nn.Module):
    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out

@dataclass
class Configuration:
    def __init__(self):
        parser = argparse.ArgumentParser(description='Train and Test on University-1652')

        # Added for your modification
        parser.add_argument('--model', default='8', type=str, help='backbone model')
        parser.add_argument('--handcraft_model', default=True, type=bool, help='use modified backbone')
        parser.add_argument('--img_size', default=384, type=int, help='input image size')
        parser.add_argument('--height', default=384, type=int, help='input image height')
        parser.add_argument('--width', default=384, type=int, help='input image width')
        parser.add_argument('--views', default=2, type=int, help='only supports 2 branches retrieval')
        parser.add_argument('--record', default=True, type=bool, help='use tensorboard to record training procedure')

        # Model Config
        parser.add_argument('--nclasses', default=768, type=int, help='1652场景的类别数')
        parser.add_argument('--block', default=1, type=int)
        parser.add_argument('--triplet_loss', default=0.3, type=float)
        parser.add_argument('--resnet', default=False, type=bool)

        # Our tricks
        # --
        parser.add_argument('--only_test', default=False, type=bool, help='use pretrained model to test')
        parser.add_argument('--ckpt_path',default='checkpoints/university/8',
                            type=str, help='path to pretrained checkpoint file')
        parser.add_argument('--data_folder', default='DMNIL/dataset', type=str)
        # Training Config
        parser.add_argument('--mixed_precision', default=True, type=bool)
        parser.add_argument('--custom_sampling', default=True, type=bool)
        parser.add_argument('--seed', default=1, type=int, help='random seed')
        parser.add_argument('--epochs', default=40, type=int, help='1 epoch for 1652')
        parser.add_argument('--batch_size', default=64, type=int, help='remember the bs is for 2 branches')
        parser.add_argument('--verbose', default=True, type=bool)
        parser.add_argument('--gpu_ids', default=(0,1,2,3), type=tuple)
        parser.add_argument('--num-instances', type=int, default=4,
                            help="each minibatch consist of ""(batch_size // num_instances) identities, and ""each identity has num_instances instances, ""default: 0 (NOT USE)")
        # Eval Config
        parser.add_argument('--batch_size_eval', default=128, type=int)
        parser.add_argument('--eval_every_n_epoch', default=1, type=int)
        parser.add_argument('--normalize_features', default=True, type=bool)
        parser.add_argument('--eval_gallery_n', default=-1, type=int)

        # Optimizer Config
        parser.add_argument('--clip_grad', default=100.0, type=float)
        parser.add_argument('--decay_exclue_bias', default=False, type=bool)
        parser.add_argument('--grad_checkpointing', default=False, type=bool)

        # Loss Config
        parser.add_argument('--label_smoothing', default=0.1, type=float)

        # Learning Rate Config
        parser.add_argument('--lr', default=0.0010, type=float, help='1 * 10^-4 for ViT | 1 * 10^-1 for CNN')
        parser.add_argument('--scheduler', default="cosine", type=str, help=r'"polynomial" | "cosine" | "constant" | None')
        parser.add_argument('--warmup_epochs', default=0.1, type=float)
        parser.add_argument('--lr_end', default=0.0001, type=float)
        parser.add_argument('--temp', type=float, default=0.05,help="temperature for scaling contrastive loss")
        # Learning part Config
        parser.add_argument('--lr_mlp', default=None, type=float)
        parser.add_argument('--lr_decouple', default=None, type=float)
        parser.add_argument('--lr_blockweights', default=2, type=float)
        parser.add_argument('--weight-decay', type=float, default=5e-4)
        parser.add_argument('--step-size', type=int, default=20)
        # Dataset Config
        parser.add_argument('--dataset', default='U1652-D2S', type=str, help="'U1652-D2S' | 'U1652-S2D'")
        parser.add_argument('--data_dir', default='/home/hk/PAPER/UCRVS/DMNIL/dataset', type=str)
        parser.add_argument('--dataset_name', default='U1652', type=str)
        parser.add_argument('--iters', type=int, default=400)
        # Augment Images Config
        parser.add_argument('--prob_flip', default=0.5, type=float, help='flipping the sat image and drone image simultaneously')
        parser.add_argument('--k1', type=int, default=30, help="hyperparameter for jaccard distance")
        parser.add_argument('--k2', type=int, default=6,  help="hyperparameter for jaccard distance")
        # Savepath for model checkpoints Config
        parser.add_argument('--model_path', default='./checkpoints/university', type=str)
        # cluster
        parser.add_argument('--eps', type=float, default=0.6,help="max neighbor distance for DBSCAN")
        parser.add_argument('--eps-gap', type=float, default=0.02, help="multi-scale criterion for measuring cluster reliability")
        parser.add_argument('--features', type=int, default=0)
        parser.add_argument('--dropout', type=float, default=0)
        parser.add_argument('--momentum', type=float, default=0.2,help="update momentum for the hybrid memory")
        parser.add_argument('--print-freq', type=int, default=10)
        # Eval before training Config
        parser.add_argument('--zero_shot', default=False, type=bool)
        parser.add_argument('--pooling-type', type=str, default='gem')
        parser.add_argument('--use-hard', action="store_true")
        parser.add_argument('--no-cam', action="store_true")
        parser.add_argument('--warmup-step', type=int, default=0)
        parser.add_argument('--milestones', nargs='+', type=int, default=[20, 40],help='milestones for the learning rate decay')
        # Checkpoint to start from Config
        parser.add_argument('--checkpoint_start', default=None)

        # Set num_workers to 0 if on Windows Config
        parser.add_argument('--num_workers', default=4 if os.name == 'nt' else 4, type=int)

        # Train on GPU if available Config
        parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', type=str)

        # For better performance Config
        parser.add_argument('--cudnn_benchmark', default=True, type=bool)

        # Make cudnn deterministic Config
        parser.add_argument('--cudnn_deterministic', default=False, type=bool)

        args = parser.parse_args(namespace=self)


# -----------------------------------------------------------------------------#
# Train Config                                                                #
# -----------------------------------------------------------------------------#
config = Configuration()

def main(config):
    best_score = 0
    start_time = time.monotonic()
    sat_batch=64
    dro_batch=64
    global start_epoch, best_mAP
    model_path = "{}/{}".format(config.model_path,
                                   config.model
                                   )
#time.strftime("%m%d%H%M%S")
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    shutil.copyfile(os.path.basename(__file__), "{}/train.py".format(model_path))

    # Redirect print to both console and log file
    sys.stdout = Logger(os.path.join(model_path, 'log.txt'))
    setup_system(seed=config.seed,
                 cudnn_benchmark=config.cudnn_benchmark,
                 cudnn_deterministic=config.cudnn_deterministic)
    if config.handcraft_model is not True:
        print("\nModel: {}".format(config.model))
        model = TimmModel(config.model,
                          pretrained=True,
                          img_size=config.img_size)
        total_params = sum(p.numel() for p in model.parameters())
        params = total_params / 1_000_000
        print(f"total parameters in model: {params:.2f}M")
    else:
        from DMNIL.hand_convnext.model import make_model

        model = make_model(config)
        print("\nModel:{}".format("adjust model: handcraft convnext-tiny"))
        total_params = sum(p.numel() for p in model.parameters())
        params = total_params / 1_000_000
        print(f"total parameters in model: {params:.2f}M")

    # Activate gradient checkpointing
    if config.grad_checkpointing:
        model.set_grad_checkpointing(True)

    # Load pretrained Checkpoint
    if config.checkpoint_start is not None:
        print("Start from:", config.checkpoint_start)
        model_state_dict = torch.load(config.checkpoint_start)
        model.load_state_dict(model_state_dict, strict=False)

        # Data parallel
    print("GPUs available:", torch.cuda.device_count())
    if torch.cuda.device_count() > 1 and len(config.gpu_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=config.gpu_ids)

    # Model to device
    model = model.to(config.device)
    data_config = model.module.get_config()
    mean = data_config["mean"]
    std = data_config["std"]
    img_size = (config.height, config.width)
    print("\nImage Size Query:", img_size)
    print("Image Size Ground:", img_size)
    print("Mean: {}".format(mean))
    print("Std:  {}\n".format(std))
    if config.dataset == 'U1652-D2S':
        config.query_folder_train = f'{config.data_folder}/{config.dataset_name}/train/satellite'
        config.gallery_folder_train = f'{config.data_folder}/{config.dataset_name}/train/drone'
        config.query_folder_test = f'{config.data_folder}/{config.dataset_name}/test/query_drone'
        config.gallery_folder_test = f'{config.data_folder}/{config.dataset_name}/test/gallery_satellite'
    elif config.dataset == 'U1652-S2D':
        config.query_folder_train = f'{config.data_folder}/{config.dataset_name}/train/satellite'
        config.gallery_folder_train = f'{config.data_folder}/{config.dataset_name}/train/drone'
        config.query_folder_test = f'{config.data_folder}/{config.dataset_name}/test/query_satellite'
        config.gallery_folder_test = f'{config.data_folder}/{config.dataset_name}/test/gallery_drone'

    val_transforms, train_sat_transforms, train_drone_transforms = get_transforms(img_size, mean=mean, std=std)
    query_dataset_test = U1652DatasetEval(data_folder=config.query_folder_test,
                                          mode="query",
                                          transforms=val_transforms,
                                          )

    query_dataloader_test = DataLoader(query_dataset_test,
                                       batch_size=config.batch_size_eval,
                                       num_workers=config.num_workers,
                                       shuffle=False,
                                       pin_memory=True)

    # Query Ground Images Test
    gallery_dataset_test = U1652DatasetEval(data_folder=config.gallery_folder_test,
                                            mode="gallery",
                                            transforms=val_transforms,
                                            sample_ids=query_dataset_test.get_sample_ids(),
                                            gallery_n=config.eval_gallery_n,
                                            )

    gallery_dataloader_test = DataLoader(gallery_dataset_test,
                                         batch_size=config.batch_size_eval,
                                         num_workers=config.num_workers,
                                         shuffle=False,
                                         pin_memory=True)

    print("Query Images Test:", len(query_dataset_test))
    print("Gallery Images Test:", len(gallery_dataset_test))

    if config.only_test:
        print("\n{}[{}]{}".format(30 * "-", "Evaluate", 30 * "-"))
        best_score = 0

        checkpoint = torch.load(config.ckpt_path)
        print(checkpoint.keys())

        state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint

        # 动态处理键名
        model_is_parallel = isinstance(model, torch.nn.DataParallel)
        weights_are_parallel = any(k.startswith('module.') for k in state_dict.keys())

        if model_is_parallel and not weights_are_parallel:

            state_dict = {f"module.{k}": v for k, v in state_dict.items()}
        elif not model_is_parallel and weights_are_parallel:

            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

        # 加载权重
        load_result = model.load_state_dict(state_dict, strict=False)

        r1_test = evaluate(config=config,
                           model=model,
                           query_loader=query_dataloader_test,
                           gallery_loader=gallery_dataloader_test,
                           ranks=[1, 5, 10],
                           step_size=1000,
                           cleanup=True)

        sys.exit()
# -----------------------------------------------------------------------------#
# Create datasets
# -----------------------------------------------------------------------------#
    iters = config.iters if (config.iters > 0) else None

    print("==> Load unlabeled dataset")

    dataset_dro = get_data('U1652_dor',config.data_dir)
    dataset_sat = get_data('U1652_sat', config.data_dir)
    test_loader_dro = get_test_loader(dataset_dro, config.height, config.width, config.batch_size, config.num_workers)
    test_loader_sat = get_test_loader(dataset_sat, config.height, config.width, config.batch_size, config.num_workers)


    # -----------------------------------------------------------------------------#
    # DataLoader                                                                  #
    # -----------------------------------------------------------------------------#

    # Transforms


    trainer = Trainer_DMNIL(model)
    trainer.cmlabel=20#30
    trainer.hm = 1#20
    trainer.ht = 10#10#10#

    params = [{"params": [value]} for _, value in model.named_parameters() if value.requires_grad]
    optimizer = torch.optim.SGD(params, lr=config.lr, momentum=0.9, weight_decay=config.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config.step_size, gamma=0.1)


    @torch.no_grad()
    def generate_cluster_features(labels, features):
        centers = collections.defaultdict(list)
        for i, label in enumerate(labels):
            if label == -1:
                continue
            centers[labels[i]].append(features[i])

        centers = [
            torch.stack(centers[idx], dim=0).mean(0) for idx in sorted(centers.keys())
        ]

        centers = torch.stack(centers, dim=0)
        return centers

    color_aug = T.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5)

    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    height = config.height
    width = config.width
    train_transformer_sat = T.Compose([
        color_aug,
        T.Resize((height, width)),  # , interpolation=3
        T.Pad(10),
        T.RandomCrop((height, width)),
        T.RandomHorizontalFlip(p=0.5),
        T.ToTensor(),
        normalizer,
    ])

    train_transformer_sat1 = T.Compose([
        color_aug,
        # T.Grayscale(num_output_channels=3),
        T.Resize((height, width)),
        T.Pad(10),
        T.RandomCrop((height, width)),
        T.RandomHorizontalFlip(p=0.5),
        T.ToTensor(),
        normalizer
    ])

    transform_dro = T.Compose([
        color_aug,
        T.Resize((height, width)),
        T.Pad(10),
        T.RandomCrop((height, width)),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        normalizer,
    ])
    transform_dro1 = T.Compose([
        color_aug,
        T.Resize((height, width)),
        T.Pad(10),
        T.RandomCrop((height, width)),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        normalizer])
    sat_cluster_num = {}
    dro_cluster_num = {}
    lenth_ratio=0
    lam=0.5
    dro_cluster_num = {}
    sat_cluster_num = {}
    for epoch in range(config.epochs):

        print(epoch)
        if (epoch == trainer.cmlabel):

            checkpoint_path,latest_epoch = load_latest_checkpoint(config)
            checkpoint = torch.load(checkpoint_path)
            print(f"Loaded checkpoint from epoch {latest_epoch}")

            state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
            model_is_parallel = isinstance(model, torch.nn.DataParallel)  # 判断模型是否是多 GPU 模式
            weights_are_parallel = any(k.startswith('module.') for k in state_dict.keys())  # 判断权重是否带 module.

            if model_is_parallel and not weights_are_parallel:

                state_dict = {f"module.{k}": v for k, v in state_dict.items()}
            elif not model_is_parallel and weights_are_parallel:

                state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}


            load_result = model.load_state_dict(state_dict, strict=False)

        with torch.no_grad():
            dro_eps = 0.40
            sat_eps = 0.30
            print('dro Clustering criterion: eps: {:.3f}'.format(dro_eps))
            cluster_dro = DBSCAN(eps=dro_eps, min_samples=4, metric='precomputed', n_jobs=-1)
            print('sat Clustering criterion: eps: {:.3f}'.format(sat_eps))
            cluster_sat = DBSCAN(eps=sat_eps, min_samples=4, metric='precomputed', n_jobs=-1)
            print('==> Create pseudo labels for unlabeled Sat data')
            cluster_loader_sat = get_test_loader(dataset_sat, config.height, config.width,
                                                 256, config.num_workers,
                                                 testset=sorted(dataset_sat.train))

            features_sat, features_sat_s = extract_features(model, cluster_loader_sat, print_freq=50)
            features_sat_s = torch.cat([features_sat_s[f].unsqueeze(0) for f, _, _ in sorted(dataset_sat.train)], 0)

            del cluster_loader_sat,
            features_sat = torch.cat([features_sat[f].unsqueeze(0) for f, _, _ in sorted(dataset_sat.train)], 0)

            features_sat_ori = features_sat
            features_sat_s_ = F.normalize(features_sat_s, dim=1)
            features_sat_ori_ = F.normalize(features_sat_ori, dim=1)
            features_sat = torch.cat((features_sat, features_sat_s), 1)
            features_sat_ = F.normalize(features_sat, dim=1)

            print('==> Create pseudo labels for unlabeled dro data')
            cluster_loader_dro = get_test_loader(dataset_dro, config.height, config.width,
                                                256, config.num_workers,
                                                testset=sorted(dataset_dro.train))

            features_dro, features_dro_s = extract_features(model, cluster_loader_dro, print_freq=50)
            del cluster_loader_dro
            features_dro = torch.cat([features_dro[f].unsqueeze(0) for f, _, _ in sorted(dataset_dro.train)], 0)

            features_dro_ori = features_dro

            features_dro_s = torch.cat([features_dro_s[f].unsqueeze(0) for f, _, _ in sorted(dataset_dro.train)], 0)

            features_dro_s_ = F.normalize(features_dro_s, dim=1)

            features_dro = torch.cat((features_dro, features_dro_s), 1)
            features_dro_ = F.normalize(features_dro, dim=1)
            features_dro_ori_ = F.normalize(features_dro_ori, dim=1)
            all_feature = []
            rerank_dist_dro = compute_jaccard_distance(features_dro_, k1=30, k2=config.k2, search_option=3)
            pseudo_labels_dro = cluster_dro.fit_predict(rerank_dist_dro)

            rerank_dist_sat = compute_jaccard_distance(features_sat_, k1=config.k1, k2=config.k2, search_option=3)
            pseudo_labels_sat = cluster_sat.fit_predict(rerank_dist_sat)

            del rerank_dist_sat
            del rerank_dist_dro

            num_cluster_dro = len(set(pseudo_labels_dro)) - (1 if -1 in pseudo_labels_dro else 0)
            num_cluster_sat = len(set(pseudo_labels_sat)) - (1 if -1 in pseudo_labels_sat else 0)
        cluster_features_dro = generate_cluster_features(pseudo_labels_dro, features_dro_ori)
        cluster_features_sat = generate_cluster_features(pseudo_labels_sat, features_sat_ori)


        if epoch >= 1:
            config.momentum = 0.1
        print('config.momentum', config.momentum)
        memory_dro = ClusterMemory(768, num_cluster_dro, temp=config.temp,
                                      momentum=config.momentum, use_hard=config.use_hard).cuda()
        memory_sat = ClusterMemory(768, num_cluster_sat, temp=config.temp,
                                       momentum=config.momentum, use_hard=config.use_hard).cuda()
        memory_dro.features = F.normalize(cluster_features_dro, dim=1).cuda()
        memory_sat.features = F.normalize(cluster_features_sat, dim=1).cuda()

        trainer.memory_dro = memory_dro
        trainer.memory_sat = memory_sat
        wise_momentum = 0.9
        print('wise_momentum', wise_momentum)
        wise_memory_sat = ClusterMemoryV2(768, len(dataset_sat.train), num_cluster_sat, temp=config.temp,
                                             momentum=wise_momentum).cuda()  # config.momentum
        wise_memory_dro = ClusterMemoryV2(768, len(dataset_dro.train), num_cluster_dro, temp=config.temp,
                                            momentum=wise_momentum).cuda()
        wise_memory_dro.features = F.normalize(features_dro_ori, dim=1).cuda()
        wise_memory_sat.features = F.normalize(features_sat_ori, dim=1).cuda()

        nameMap_dro = {val[0]: idx for (idx, val) in enumerate(sorted(dataset_dro.train))}

        nameMap_sat = {val[0]: idx for (idx, val) in enumerate(sorted(dataset_sat.train))}

        wise_memory_sat.labels = torch.from_numpy(pseudo_labels_sat)
        wise_memory_dro.labels = torch.from_numpy(pseudo_labels_dro)

        trainer.wise_memory_dro = wise_memory_dro
        trainer.wise_memory_sat = wise_memory_sat
        trainer.nameMap_dro = nameMap_dro
        trainer.nameMap_sat = nameMap_sat

        cluster_features_dro_s = generate_cluster_features(pseudo_labels_dro, features_dro_s)
        cluster_features_sat_s = generate_cluster_features(pseudo_labels_sat, features_sat_s)

        memory_dro_s = DHML(768, num_cluster_dro, temp=config.temp,
                                        momentum=config.momentum, use_hard=config.use_hard).cuda()
        memory_sat_s = DHML(768, num_cluster_sat, temp=config.temp,
                                         momentum=config.momentum, use_hard=config.use_hard).cuda()
        memory_dro_s.features = F.normalize(cluster_features_dro_s, dim=1).cuda()
        memory_sat_s.features = F.normalize(cluster_features_sat_s, dim=1).cuda()

        trainer.memory_dro_s = memory_dro_s
        trainer.memory_sat_s = memory_sat_s

        wise_memory_sat_s = ClusterMemoryV2(768, len(dataset_sat.train), num_cluster_sat, temp=config.temp,
                                               momentum=wise_momentum).cuda()  # 0.9
        wise_memory_dro_s = ClusterMemoryV2(768, len(dataset_dro.train), num_cluster_dro, temp=config.temp,
                                              momentum=wise_momentum).cuda()  # config.momentum
        wise_memory_dro_s.features = F.normalize(features_dro_s, dim=1).cuda()
        wise_memory_sat_s.features = F.normalize(features_sat_s, dim=1).cuda()
        trainer.wise_memory_dro_s = wise_memory_dro_s
        trainer.wise_memory_sat_s = wise_memory_sat_s

        pseudo_labeled_dataset_dro = []
        dro_label = []
        pseudo_real_dro = {}
        cams_dro = []
        modality_dro = []
        outlier = 0
        cross_cam = []
        idxs_dro = []
        dro_cluster = collections.defaultdict(list)

        for i, ((fname, _, cid), label) in enumerate(zip(sorted(dataset_dro.train), pseudo_labels_dro)):
            cams_dro.append(cid)
            modality_dro.append(1)
            cross_cam.append(int(cid + 4))
            dro_label.append(label.item())
            dro_cluster[cid].append(label.item())
            if label != -1:
                pseudo_labeled_dataset_dro.append((fname, label.item(), cid))

                pseudo_real_dro[label.item()] = pseudo_real_dro.get(label.item(), []) + [_]
                pseudo_real_dro[label.item()] = list(set(pseudo_real_dro[label.item()]))

            else:
                outlier = outlier + 1

        pseudo_labeled_dataset_sat = []
        sat_label = []
        pseudo_real_sat = {}
        cams_sat = []
        modality_sat = []
        outlier = 0
        idxs_sat = []
        sat_cluster = collections.defaultdict(list)

        for i, ((fname, _, cid), label) in enumerate(zip(sorted(dataset_sat.train), pseudo_labels_sat)):
            cams_sat.append(cid)
            modality_sat.append(0)
            cross_cam.append(int(cid))
            sat_label.append(label.item())
            sat_cluster[cid].append(label.item())
            if label != -1:
                pseudo_labeled_dataset_sat.append((fname, label.item(), cid))

                pseudo_real_sat[label.item()] = pseudo_real_sat.get(label.item(), []) + [_]
                pseudo_real_sat[label.item()] = list(set(pseudo_real_sat[label.item()]))
            else:
                outlier = outlier + 1

        pseudo_labels_sat_ori = torch.from_numpy(pseudo_labels_sat)

        if epoch >= trainer.cmlabel:
            with torch.no_grad():
                TOPK2 = 10
                Score_TOPK = 10
                features_sat_s_ = features_sat_ori_ + torch.randn_like(features_sat_ori_) * 0.05
                features_dro_s_ = features_dro_ori_ + torch.randn_like(features_dro_ori_) * 0.05
                cluster_label_dro_self = trainer.wise_memory_dro.labels.detach().cpu()
                ins_sim_sat_dro = features_sat_ori_.mm(features_dro_ori_.t())
                ins_sim_sat_dro_s = features_sat_s_.mm(features_dro_s_.t())
                topk, ins_indices_sat_dro_s = torch.topk(ins_sim_sat_dro_s, int(Score_TOPK))  # 20
                ins_label_sat_dro = cluster_label_dro_self[
                ins_indices_sat_dro_s].detach().cpu()  # trainer.wise_memory_dro.labels cluster_label_dro_self[ins_indices_sat_dro_s].detach().cpu()#.numpy()#.view(-1)
                topk, ins_indices_sat_dro = torch.topk(ins_sim_sat_dro, int(Score_TOPK))  # 20
                cluster_label_sat_dro = cluster_label_dro_self[
                ins_indices_sat_dro].detach().cpu()  # .numpy()#.view(-1)
                intersect_count_list = []
                for l in range(TOPK2):
                    intersect_count = (ins_label_sat_dro == cluster_label_sat_dro[:, l].view(-1, 1)).int().sum(
                            1).view(-1, 1).detach().cpu()
                    intersect_count_list.append(intersect_count)

                intersect_count_list = torch.cat(intersect_count_list, 1)
                intersect_count, _ = intersect_count_list.max(1)
                topk, cluster_label_index = torch.topk(intersect_count_list, 1)
                cluster_label_sat_dro = torch.gather(cluster_label_sat_dro, dim=1, index=cluster_label_index.view(-1,
                                                                                                                    1)).cpu().numpy()  # cluster_label_sat_dro[cluster_label_index.reshape(-1,1)]
                cluster_label_sat_dro = torch.from_numpy(cluster_label_sat_dro)
                print('soft structure smooth v3')
                sat_cm_label = cluster_label_sat_dro.view(-1) + 1
                lp_feat_sat = features_sat_ori_
                lp_feat_sat_s = features_sat_s_

                sat_cm_label = F.one_hot(sat_cm_label.view(lp_feat_sat.size(0), 1).long(),
                                             int(num_cluster_dro) + 1).float().squeeze(1)

                sat_self_sim = torch.mm(lp_feat_sat, lp_feat_sat.t())
                sat_self_sim_s = torch.mm(lp_feat_sat_s, lp_feat_sat_s.t())

                sat_self_sim = sat_self_sim + sat_self_sim_s

                topk_self, indices_self = torch.topk(sat_self_sim, 5)  # 20
                mask_self = torch.zeros_like(sat_self_sim)
                mask_self = mask_self.scatter(1, indices_self, 1)
                sat_self_sim = mask_self

                smooth_sat = torch.mm(sat_self_sim.cpu(), sat_cm_label.cpu())
                smooth_sat = torch.argmax(smooth_sat, 1).view(-1).numpy()
                pseudo_labels_sat_cm = [int(smolabel - 1) for smolabel in smooth_sat]
                pseudo_labels_sat_cm = np.array(pseudo_labels_sat_cm)
                cluster_label_sat_dro = torch.from_numpy(pseudo_labels_sat_cm)  # .view(-1)

                del sat_self_sim, smooth_sat, lp_feat_sat, lp_feat_sat_s

            pseudo_labels_sat = cluster_label_sat_dro.view(-1).cpu().numpy()

            num_cluster_sat = len(set(pseudo_labels_sat)) - (1 if -1 in pseudo_labels_sat else 0)
            num_cluster_dro = len(set(pseudo_labels_dro)) - (1 if -1 in pseudo_labels_dro else 0)

            pseudo_labeled_dataset_dro = []
            cams_dro = []
            modality_dro = []
            cross_cam = []
            for i, ((fname, _, cid), label) in enumerate(zip(sorted(dataset_dro.train), pseudo_labels_dro)):
                cams_dro.append(int(cid + 4))
                modality_dro.append(int(1))
                cross_cam.append(int(cid + 4))
                indexes = torch.tensor([trainer.nameMap_dro[fname]])
                dro_label_ms = trainer.wise_memory_dro.labels[indexes]

                if (label != -1) and (dro_label_ms != -1):
                    pseudo_labeled_dataset_dro.append((fname, label.item(), cid))
                    # if epoch%10 == 0:
                    #     print(fname,label.item())

            pseudo_labeled_dataset_sat = []
            cams_sat = []
            modality_sat = []
            for i, ((fname, _, cid), label) in enumerate(zip(sorted(dataset_sat.train), pseudo_labels_sat)):
                cams_sat.append(int(cid))
                modality_sat.append(int(0))
                cross_cam.append(int(cid))
                indexes = torch.tensor([trainer.nameMap_sat[fname]])
                sat_label_ms = trainer.wise_memory_sat.labels[indexes]

                if (label != -1) and (sat_label_ms != -1):
                    pseudo_labeled_dataset_sat.append((fname, label.item(), cid))


            features_all = torch.cat((features_sat_ori, features_dro_ori), dim=0)

            pseudo_labels_all = torch.cat((torch.from_numpy(pseudo_labels_sat), torch.from_numpy(pseudo_labels_dro)),
                                              dim=-1).view(-1).cpu().numpy()

            cluster_features_dro = generate_cluster_features(pseudo_labels_all, features_all)

            shared_memory = ClusterMemory(768, num_cluster_dro, temp=config.temp,
                                              momentum=0.1, use_hard=config.use_hard)  # .cuda()
            shared_memory.features = F.normalize(cluster_features_dro, dim=1).cuda()

            trainer.memory_dro = shared_memory
            trainer.memory_sat = shared_memory
            features_all_s = torch.cat((features_sat_s, features_dro_s), dim=0)
            cluster_features_dro_s = generate_cluster_features(pseudo_labels_all, features_all_s)
            shared_memory_s = ClusterMemory(768, num_cluster_dro, temp=config.temp,
                                                momentum=0.1, use_hard=config.use_hard)

            shared_memory_s.features = F.normalize(cluster_features_dro_s, dim=1).cuda()

            trainer.memory_sat_s = shared_memory_s
            trainer.memory_dro_s = shared_memory_s

        train_loader_dro = get_train_loader_dro(config, dataset_dro, config.height, config.width,
                                                  dro_batch, config.num_workers, config.num_instances, iters,
                                                  trainset=pseudo_labeled_dataset_dro, no_cam=config.no_cam,
                                                  train_transformer= transform_dro)
        train_loader_sat = get_train_loader_sat(config, dataset_sat, config.height, config.width,
                                                      sat_batch, config.num_workers, config.num_instances, iters,
                                                      trainset=pseudo_labeled_dataset_sat, no_cam=config.no_cam,
                                                      train_transformer= train_transformer_sat,
                                                      train_transformer1= train_transformer_sat1)

        train_loader_dro.new_epoch()
        train_loader_sat.new_epoch()
        trainer.train(epoch, train_loader_dro, train_loader_sat, optimizer, print_freq=config.print_freq,
                          train_iters=len(train_loader_dro))

        # evaluate
        if (epoch % config.eval_every_n_epoch == 0):

            print("\n{}[{}]{}".format(30 * "-", "Evaluate", 30 * "-"))

            r1_test = evaluate(config=config,
                               model=model,
                               query_loader=query_dataloader_test,
                               gallery_loader=gallery_dataloader_test,
                               ranks=[1, 5, 10],
                               step_size=1000,
                               cleanup=True)

            if r1_test > best_score:

                best_score = r1_test

                if torch.cuda.device_count() > 1 and len(config.gpu_ids) > 1:
                    torch.save(model.module.state_dict(),
                               '{}/weights_e{}.pth'.format(model_path, epoch))
                else:
                    torch.save(model.state_dict(), '{}/weights_e{}_{:.4f}.pth'.format(model_path, epoch, r1_test))

        lr_scheduler.step()
    end_time = time.monotonic()
    print('Total running time: ', timedelta(seconds=end_time - start_time))
if __name__ == "__main__":
    main(config)
