import os
import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset
import copy
from tqdm import tqdm
import time
import random
from albumentations.core.transforms_interface import ImageOnlyTransform
import imgaug.augmenters as iaa


def get_data(path):
    data = {}
    for root, dirs, files in os.walk(path, topdown=False):
        for name in dirs:
            data[name] = {"path": os.path.join(root, name)}
            for _, _, files in os.walk(data[name]["path"], topdown=False):
                data[name]["files"] = files

    return data


class U1652DatasetTrain(Dataset):

    def __init__(self,
                 query_folder,
                 gallery_folder,
                 transforms_query=None,
                 transforms_gallery=None,
                 prob_flip=0.5,
                 shuffle_batch_size=128):
        super().__init__()

        self.query_dict = get_data(query_folder)
        self.gallery_dict = get_data(gallery_folder)

        # use only folders that exists for both gallery and query
        self.ids = list(set(self.query_dict.keys()).intersection(self.gallery_dict.keys()))
        self.ids.sort()
        self.map_dict = {i: self.ids[i] for i in range(len(self.ids))}
        self.reverse_map_dict = {v: k for k, v in self.map_dict.items()}

        self.pairs = []

        for idx in self.ids:

            query_img = "{}/{}".format(self.query_dict[idx]["path"],
                                       self.query_dict[idx]["files"][0])

            gallery_path = self.gallery_dict[idx]["path"]
            gallery_imgs = self.gallery_dict[idx]["files"]

            label = self.reverse_map_dict[idx]

            for g in gallery_imgs:
                self.pairs.append((idx, label, query_img, "{}/{}".format(gallery_path, g)))

        self.transforms_query = transforms_query
        self.transforms_gallery = transforms_gallery
        self.prob_flip = prob_flip
        self.shuffle_batch_size = shuffle_batch_size

        self.samples = copy.deepcopy(self.pairs)

    def __getitem__(self, index):

        idx, label, query_img_path, gallery_img_path = self.samples[index]

        # for query there is only one file in folder
        query_img = cv2.imread(query_img_path)
        query_img = cv2.cvtColor(query_img, cv2.COLOR_BGR2RGB)

        gallery_img = cv2.imread(gallery_img_path)
        gallery_img = cv2.cvtColor(gallery_img, cv2.COLOR_BGR2RGB)

        if np.random.random() < self.prob_flip:
            query_img = cv2.flip(query_img, 1)
            gallery_img = cv2.flip(gallery_img, 1)

            # image transforms
        if self.transforms_query is not None:
            query_img = self.transforms_query(image=query_img)['image']

        if self.transforms_gallery is not None:
            gallery_img = self.transforms_gallery(image=gallery_img)['image']

        return query_img, gallery_img, idx, label

    def __len__(self):
        return len(self.samples)

    def shuffle(self, ):

        '''
        custom shuffle function for unique class_id sampling in batch
        '''

        print("\nShuffle Dataset:")

        pair_pool = copy.deepcopy(self.pairs)

        # Shuffle pairs order
        random.shuffle(pair_pool)

        # Lookup if already used in epoch
        pairs_epoch = set()
        idx_batch = set()

        # buckets
        batches = []
        current_batch = []

        # counter
        break_counter = 0

        # progressbar
        pbar = tqdm()

        while True:

            pbar.update()

            if len(pair_pool) > 0:
                pair = pair_pool.pop(0)

                idx, _, _, _ = pair

                if idx not in idx_batch and pair not in pairs_epoch:

                    idx_batch.add(idx)
                    current_batch.append(pair)
                    pairs_epoch.add(pair)

                    break_counter = 0

                else:
                    # if pair fits not in batch and is not already used in epoch -> back to pool
                    if pair not in pairs_epoch:
                        pair_pool.append(pair)

                    break_counter += 1

                if break_counter >= 512:
                    break

            else:
                break

            if len(current_batch) >= self.shuffle_batch_size:
                # empty current_batch bucket to batches
                batches.extend(current_batch)
                idx_batch = set()
                current_batch = []

        pbar.close()

        # wait before closing progress bar
        time.sleep(0.3)

        self.samples = batches

        print("Original Length: {} - Length after Shuffle: {}".format(len(self.pairs), len(self.samples)))
        print("Break Counter:", break_counter)
        print("Pairs left out of last batch to avoid creating noise:", len(self.pairs) - len(self.samples))
        print("First Element ID: {} - Last Element ID: {}".format(self.samples[0][0], self.samples[-1][0]))


class U1652DatasetEval(Dataset):

    def __init__(self,
                 data_folder,
                 mode,
                 transforms=None,
                 sample_ids=None,
                 gallery_n=-1):
        super().__init__()

        self.data_dict = get_data(data_folder)

        # use only folders that exists for both gallery and query
        self.ids = list(self.data_dict.keys())

        self.transforms = transforms

        self.given_sample_ids = sample_ids

        self.images = []
        self.sample_ids = []

        self.mode = mode

        self.gallery_n = gallery_n

        for i, sample_id in enumerate(self.ids):

            for j, file in enumerate(self.data_dict[sample_id]["files"]):
                self.images.append("{}/{}".format(self.data_dict[sample_id]["path"],
                                                  file))

                self.sample_ids.append(sample_id)

    def __getitem__(self, index):

        img_path = self.images[index]
        sample_id = self.sample_ids[index]

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # if self.mode == "sat":

        #    img90 = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        #    img180 = cv2.rotate(img90, cv2.ROTATE_90_CLOCKWISE)
        #    img270 = cv2.rotate(img180, cv2.ROTATE_90_CLOCKWISE)

        #    img_0_90 = np.concatenate([img, img90], axis=1)
        #    img_180_270 = np.concatenate([img180, img270], axis=1)

        #    img = np.concatenate([img_0_90, img_180_270], axis=0)

        # image transforms
        if self.transforms is not None:
            img = self.transforms(image=img)['image']

        label = int(sample_id)
        if self.given_sample_ids is not None:
            if sample_id not in self.given_sample_ids:
                label = -1

        return img, label , img_path

    def __len__(self):
        return len(self.images)

    def get_sample_ids(self):
        return set(self.sample_ids)


# ********************************************** apply multi-weather setting for U1652 **********************************************************

class ImgAugTransform(ImageOnlyTransform):
    def __init__(self, aug, always_apply=False, p=1.0):
        super(ImgAugTransform, self).__init__(always_apply, p)
        self.aug = aug

    def apply(self, img, **params):
        return self.aug(image=img)


# 自定义云层变换
class CustomCloudLayer(ImgAugTransform):
    def __init__(self, intensity_mean=225, intensity_freq_exponent=-2, intensity_coarse_scale=2,
                 alpha_min=1.0, alpha_multiplier=0.9, alpha_size_px_max=10, alpha_freq_exponent=-2,
                 sparsity=0.9, density_multiplier=0.5, seed=None, always_apply=False, p=1.0):
        aug = iaa.CloudLayer(
            intensity_mean=intensity_mean,
            intensity_freq_exponent=intensity_freq_exponent,
            intensity_coarse_scale=intensity_coarse_scale,
            alpha_min=alpha_min,
            alpha_multiplier=alpha_multiplier,
            alpha_size_px_max=alpha_size_px_max,
            alpha_freq_exponent=alpha_freq_exponent,
            sparsity=sparsity,
            density_multiplier=density_multiplier,
            seed=seed
        )
        super(CustomCloudLayer, self).__init__(aug, always_apply, p)


# 自定义雨变换
class CustomRain(ImgAugTransform):
    def __init__(self, drop_size=(0.05, 0.1), speed=(0.04, 0.06), seed=None, always_apply=False, p=1.0):
        aug = iaa.Rain(
            drop_size=drop_size,
            speed=speed,
            seed=seed
        )
        super(CustomRain, self).__init__(aug, always_apply, p)


# 自定义雪花变换
class CustomSnowflakes(ImgAugTransform):
    def __init__(self, flake_size=(0.5, 0.8), speed=(0.007, 0.03), seed=None, always_apply=False, p=1.0):
        aug = iaa.Snowflakes(
            flake_size=flake_size,
            speed=speed,
            seed=seed
        )
        super(CustomSnowflakes, self).__init__(aug, always_apply, p)


iaa_weather_list = [

    # 0. Normal
    A.NoOp(),
    # 1. Fog
    A.Compose([
        CustomCloudLayer()
    ]),
    # 2. Rain
    A.Compose([
        CustomRain(drop_size=(0.05, 0.1), speed=(0.04, 0.06), seed=38),
        CustomRain(drop_size=(0.05, 0.1), speed=(0.04, 0.06), seed=35),
        CustomRain(drop_size=(0.1, 0.2), speed=(0.04, 0.06), seed=73),
        CustomRain(drop_size=(0.1, 0.2), speed=(0.04, 0.06), seed=93),
        CustomRain(drop_size=(0.05, 0.2), speed=(0.04, 0.06), seed=95),
    ]),
    # 3. Snow
    A.Compose([
        CustomSnowflakes(flake_size=(0.5, 0.8), speed=(0.007, 0.03), seed=38),
        CustomSnowflakes(flake_size=(0.5, 0.8), speed=(0.007, 0.03), seed=35),
        CustomSnowflakes(flake_size=(0.6, 0.9), speed=(0.007, 0.03), seed=74),
        CustomSnowflakes(flake_size=(0.6, 0.9), speed=(0.007, 0.03), seed=94),
        CustomSnowflakes(flake_size=(0.5, 0.9), speed=(0.007, 0.03), seed=96),
    ]),
    # 4. Fog+Rain
    A.Compose([
        CustomCloudLayer(),
        CustomRain(drop_size=(0.05, 0.2), speed=(0.04, 0.06), seed=35),
        CustomRain(drop_size=(0.05, 0.2), speed=(0.04, 0.06), seed=36)
    ]),
    # 5. Fog+Snow
    A.Compose([
        CustomCloudLayer(),
        CustomSnowflakes(flake_size=(0.5, 0.9), speed=(0.007, 0.03), seed=35),
        CustomSnowflakes(flake_size=(0.5, 0.9), speed=(0.007, 0.03), seed=36)
    ]),
    # 6. Rain+Snow
    A.Compose([
        CustomSnowflakes(flake_size=(0.5, 0.8), speed=(0.007, 0.03), seed=35),
        CustomRain(drop_size=(0.05, 0.1), speed=(0.04, 0.06), seed=35),
        CustomRain(drop_size=(0.1, 0.2), speed=(0.04, 0.06), seed=92),
        CustomRain(drop_size=(0.05, 0.2), speed=(0.04, 0.06), seed=91),
        CustomSnowflakes(flake_size=(0.6, 0.9), speed=(0.007, 0.03), seed=74),
    ]),
    # 7. Dark
    A.Compose([
        A.OneOf([
            A.GaussianBlur(blur_limit=(9, 11), p=0.5),
            A.MultiplicativeNoise(multiplier=[0.5, 1.5], per_channel=True, p=0.5),
        ]),
        A.RandomBrightnessContrast(brightness_limit=(-0.3, -0.15), contrast_limit=(0.2, 0.2), p=1)
    ]),
    # 8. Over-exposure
    A.Compose([
        A.RandomBrightnessContrast(brightness_limit=(0, 0.3), contrast_limit=(1.3, 1.6), p=1)
    ]),
    # 9. Wind
    A.Compose([
        A.MotionBlur(blur_limit=15, p=1)
    ])
]


def get_transforms(img_size,
                   mean=[0.485, 0.456, 0.406],
                   std=[0.229, 0.224, 0.225]):
    # weather_id = 0
    val_transforms = A.Compose([A.Resize(img_size[0], img_size[1], interpolation=cv2.INTER_LINEAR_EXACT, p=1.0),

                                # Multi-weather U1652 settings.
                                # iaa_weather_list[weather_id],

                                A.Normalize(mean, std),
                                ToTensorV2(),
                                ])

    train_sat_transforms = A.Compose([A.ImageCompression(quality_lower=90, quality_upper=100, p=0.5),
                                      A.Resize(img_size[0], img_size[1], interpolation=cv2.INTER_LINEAR_EXACT, p=1.0),

                                      # # Multi-weather U1652 settings.
                                      # A.OneOf(iaa_weather_list, p=1.0),

                                      A.ColorJitter(brightness=0.15, contrast=0.3, saturation=0.3, hue=0.3,
                                                    always_apply=False, p=0.5),
                                      A.OneOf([
                                          A.AdvancedBlur(p=1.0),
                                          A.Sharpen(p=1.0),
                                      ], p=0.3),
                                      A.OneOf([
                                          A.GridDropout(ratio=0.4, p=1.0),
                                          A.CoarseDropout(max_holes=25,
                                                          max_height=int(0.2 * img_size[0]),
                                                          max_width=int(0.2 * img_size[0]),
                                                          min_holes=10,
                                                          min_height=int(0.1 * img_size[0]),
                                                          min_width=int(0.1 * img_size[0]),
                                                          p=1.0),
                                      ], p=0.3),
                                      A.RandomRotate90(p=1.0),
                                      A.Normalize(mean, std),
                                      ToTensorV2(),
                                      ])

    train_drone_transforms = A.Compose([A.ImageCompression(quality_lower=90, quality_upper=100, p=0.5),
                                        A.Resize(img_size[0], img_size[1], interpolation=cv2.INTER_LINEAR_EXACT, p=1.0),

                                        # Multi-weather U1652 settings.
                                        # A.OneOf(iaa_weather_list, p=1.0),

                                        A.ColorJitter(brightness=0.15, contrast=0.7, saturation=0.3, hue=0.3,
                                                      always_apply=False, p=0.5),
                                        A.OneOf([
                                            A.AdvancedBlur(p=1.0),
                                            A.Sharpen(p=1.0),
                                        ], p=0.3),
                                        A.OneOf([
                                            A.GridDropout(ratio=0.4, p=1.0),
                                            A.CoarseDropout(max_holes=25,
                                                            max_height=int(0.2 * img_size[0]),
                                                            max_width=int(0.2 * img_size[0]),
                                                            min_holes=10,
                                                            min_height=int(0.1 * img_size[0]),
                                                            min_width=int(0.1 * img_size[0]),
                                                            p=1.0),
                                        ], p=0.3),
                                        A.Normalize(mean, std),
                                        ToTensorV2(),
                                        ])

    return val_transforms, train_sat_transforms, train_drone_transforms

class U1652DatasetTrain_unlable(Dataset):
    """
    用于无监督训练的 U1652 数据集。

    特点:
        - 支持动态生成伪标签。
        - 每个样本对 (query 和 gallery) 不依赖真实标签。
        - 支持数据增强、随机翻转等操作。
    """

    def __init__(self,
                 query_folder,
                 gallery_folder,
                 transforms_query=None,
                 transforms_gallery=None,
                 prob_flip=0.5,
                 shuffle_batch_size=128):
        """
        初始化数据集。

        参数:
            query_folder (str): Query 数据集的文件夹路径。
            gallery_folder (str): Gallery 数据集的文件夹路径。
            transforms_query (callable, optional): 对 query 图像的变换方法。
            transforms_gallery (callable, optional): 对 gallery 图像的变换方法。
            prob_flip (float, optional): 图像随机翻转的概率。
            shuffle_batch_size (int, optional): 随机打乱时每批的大小。
        """
        super().__init__()

        # 加载 Query 文件夹的所有图像路径
        self.query_images = []
        for root, _, files in os.walk(query_folder):
            for file in files:
                if file.endswith(('.jpg', '.jpeg', '.png')):
                    self.query_images.append(os.path.join(root, file))

        if not self.query_images:
            raise ValueError(f"No valid query images found in folder: {query_folder}")
        print(f"Loaded {len(self.query_images)} query images.")

        # 加载 Gallery 文件夹的所有数据
        self.gallery_dict = get_data(gallery_folder)

        # 确保 Query 和 Gallery 中都存在的目录
        self.ids = list(set(self.gallery_dict.keys()))
        self.ids.sort()

        # 初始化样本对 (Query_img, Gallery_img)
        self.pairs = []
        for idx in self.ids:
            gallery_path = self.gallery_dict[idx]["path"]
            gallery_imgs = self.gallery_dict[idx]["files"]
            for g in gallery_imgs:
                self.pairs.append((None, "{}/{}".format(gallery_path, g)))  # Query 由随机采样生成

        # 数据增强相关参数
        self.transforms_query = transforms_query
        self.transforms_gallery = transforms_gallery
        self.prob_flip = prob_flip
        self.shuffle_batch_size = shuffle_batch_size

        # 伪标签初始化为空
        self.samples = copy.deepcopy(self.pairs)
        self.pseudo_labels = None

    def update_pseudo_labels(self, pseudo_labels):
        """
        更新伪标签。

        参数:
            pseudo_labels (list): 基于相似度或其他策略生成的伪标签列表。
        """
        if len(pseudo_labels) != len(self.pairs):
            raise ValueError("伪标签的数量必须与样本对数量一致！")
        self.pseudo_labels = pseudo_labels
        self.samples = [(query, gallery, pseudo_labels[idx]) for idx, (query, gallery) in enumerate(self.pairs)]

    def shuffle(self):
        """
        自定义的随机打乱函数，确保每批数据包含唯一的类 ID。
        """
        print("\nShuffle Dataset:")

        pair_pool = copy.deepcopy(self.pairs)  # 深拷贝原始样本对以保护数据完整性
        random.shuffle(pair_pool)  # 随机打乱样本对顺序

        # 追踪本次 epoch 已使用的样本对和类 ID
        pairs_epoch = set()
        idx_batch = set()

        # 存储所有的批次
        batches = []
        current_batch = []

        # 防止死循环的计数器
        break_counter = 0

        # 进度条用于显示当前的进度
        pbar = tqdm(total=len(pair_pool), desc="Shuffling pairs", unit="pairs")

        while True:
            if len(pair_pool) > 0:
                pair = pair_pool.pop(0)  # 从池中取出一对样本
                query, gallery = pair

                if gallery not in idx_batch and pair not in pairs_epoch:
                    idx_batch.add(gallery)  # 记录当前批次中使用的类 ID
                    current_batch.append(pair)
                    pairs_epoch.add(pair)  # 记录本次 epoch 中已使用的样本对
                    break_counter = 0
                else:
                    # 如果当前样本对不合适但未被完全用过，放回池中
                    if pair not in pairs_epoch:
                        pair_pool.append(pair)
                    break_counter += 1

                # 如果长时间无法找到新的样本对，退出循环
                if break_counter >= 512:
                    break

            else:
                break

            # 如果当前批次达到指定大小，将其存入批次列表并重置
            if len(current_batch) >= self.shuffle_batch_size:
                batches.extend(current_batch)
                idx_batch = set()
                current_batch = []

            pbar.update()  # 更新进度条

        pbar.close()
        time.sleep(0.3)  # 等待进度条刷新完成

        self.samples = batches  # 更新样本
        print("Original Length: {} - Length after Shuffle: {}".format(len(self.pairs), len(self.samples)))
        print("Break Counter:", break_counter)
        print("Pairs left out of last batch to avoid noise:", len(self.pairs) - len(self.samples))

    def __getitem__(self, index):
        """
        获取指定索引的样本对 (query_img, gallery_img, pseudo_label)。

        参数:
            index (int): 索引。

        返回:
            tuple: 包含 Query 图像、Gallery 图像及伪标签的元组。
        """
        sample = self.samples[index]

        if len(sample) == 3:  # 如果样本包含伪标签
            _, gallery_img_path, pseudo_label = sample  # Query 路径被动态采样替换
        else:
            _, gallery_img_path = sample
            pseudo_label = -1  # 未指定伪标签时返回默认值

        # 随机选择一个 Query 图像
        query_img_path = random.choice(self.query_images)

        # 加载 Query 图像
        query_img = cv2.imread(query_img_path)
        query_img = cv2.cvtColor(query_img, cv2.COLOR_BGR2RGB)

        # 加载 Gallery 图像
        gallery_img = cv2.imread(gallery_img_path)
        gallery_img = cv2.cvtColor(gallery_img, cv2.COLOR_BGR2RGB)

        # 随机翻转图像
        if np.random.random() < self.prob_flip:
            query_img = cv2.flip(query_img, 1)
            gallery_img = cv2.flip(gallery_img, 1)

        # 应用数据增强
        if self.transforms_query is not None:
            query_img = self.transforms_query(image=query_img)['image']
        if self.transforms_gallery is not None:
            gallery_img = self.transforms_gallery(image=gallery_img)['image']

        return query_img, gallery_img, pseudo_label

    def __len__(self):
        """
        返回数据集的大小。

        返回:
            int: 数据集的样本数。
        """
        return len(self.samples)

from torch.utils.data import DataLoader
if __name__ == "__main__":
    # 假设你已经有了训练集和测试集的路径
    Drone_folder = 'U1652/train/drone'  # 替换为实际路径
    Satellite_folder = 'U1652/train/satellite'  # 替换为实际路径
    img_size = (384,384)
    prob_flip = 0.5
    batch_size = 2
    val_transforms, train_sat_transforms, train_drone_transforms = get_transforms(img_size)

    # 创建数据集对象
    dataset_dro = U1652DatasetTrain_unlable(
        query_folder=Drone_folder,
        gallery_folder=Satellite_folder,
        transforms_query=train_sat_transforms,
        transforms_gallery=train_drone_transforms,
        prob_flip = prob_flip,
        shuffle_batch_size = batch_size,
    )
    dataloader = DataLoader(
        dataset_dro,
        batch_size=batch_size,
        shuffle=True,  # 是否打乱数据
        num_workers=4,  # 使用多少个子进程加载数据
        pin_memory=True  # 加速数据加载到GPU
    )
    # 打印打乱后的样本数据
    print("Shuffled samples:")
    for sample in dataset_dro.samples[:20]:  # 只显示前5个样本作为例子
        print(sample)
    print("\nSample batch from DataLoader:")
    for batch in dataloader:
        images, idx, labels = batch  # 假设 batch 中包含图像数据、id 和标签
        print(f"Images batch shape: {images.shape}")
        print(f"Index batch: {idx[:5]}")  # 打印前5个 index
        print(f"Labels batch: {labels[:5]}")  # 打印前5个 labels
        break  # 只打印一个批次，若要查看更多批次，可以移除 break

