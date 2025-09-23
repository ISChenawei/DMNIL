import cv2
from albumentations.pytorch import ToTensorV2
import albumentations as A
from albumentations.core.transforms_interface import ImageOnlyTransform
import imgaug.augmenters as iaa
import numpy as np


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
    A.NoOp(),
    A.Compose([
        CustomCloudLayer()
    ]),
    A.Compose([
        CustomRain(drop_size=(0.05, 0.1), speed=(0.04, 0.06), seed=38),
        CustomRain(drop_size=(0.05, 0.1), speed=(0.04, 0.06), seed=35),
        CustomRain(drop_size=(0.1, 0.2), speed=(0.04, 0.06), seed=73),
        CustomRain(drop_size=(0.1, 0.2), speed=(0.04, 0.06), seed=93),
        CustomRain(drop_size=(0.05, 0.2), speed=(0.04, 0.06), seed=95),
    ]),
    A.Compose([
        CustomSnowflakes(flake_size=(0.5, 0.8), speed=(0.007, 0.03), seed=38),
        CustomSnowflakes(flake_size=(0.5, 0.8), speed=(0.007, 0.03), seed=35),
        CustomSnowflakes(flake_size=(0.6, 0.9), speed=(0.007, 0.03), seed=74),
        CustomSnowflakes(flake_size=(0.6, 0.9), speed=(0.007, 0.03), seed=94),
        CustomSnowflakes(flake_size=(0.5, 0.9), speed=(0.007, 0.03), seed=96),
    ]),
    A.Compose([
        A.OneOf([
            A.GaussianBlur(blur_limit=(10, 12), p=0.5),
            A.BlendAlphaOverlay(
                overlay=A.Add(value=100),
                overlay_bboxes=None,
                alpha_limit=(0.4, 0.6),
                always_apply=True
            ),
        ]),
        A.RandomBrightnessContrast(brightness_limit=(-0.3, -0.15), contrast_limit=(0.2, 0.2), p=1)
    ]),
    A.Compose([
        A.RandomBrightnessContrast(brightness_limit=(0, 0.3), contrast_limit=(1.3, 1.6), p=1)
    ]),
    A.Compose([
        CustomCloudLayer(),
        CustomRain(drop_size=(0.05, 0.2), speed=(0.04, 0.06), seed=35),
        CustomRain(drop_size=(0.05, 0.2), speed=(0.04, 0.06), seed=36)
    ]),
    A.Compose([
        CustomCloudLayer(),
        CustomSnowflakes(flake_size=(0.5, 0.9), speed=(0.007, 0.03), seed=35),
        CustomSnowflakes(flake_size=(0.5, 0.9), speed=(0.007, 0.03), seed=36)
    ]),
    A.Compose([
        CustomSnowflakes(flake_size=(0.5, 0.8), speed=(0.007, 0.03), seed=35),
        CustomRain(drop_size=(0.05, 0.1), speed=(0.04, 0.06), seed=35),
        CustomRain(drop_size=(0.1, 0.2), speed=(0.04, 0.06), seed=92),
        CustomRain(drop_size=(0.05, 0.2), speed=(0.04, 0.06), seed=91),
        CustomSnowflakes(flake_size=(0.6, 0.9), speed=(0.007, 0.03), seed=74),
    ]),
    A.Compose([
        A.MotionBlur(blur_limit=15, p=1)
    ])
]

class Cut(ImageOnlyTransform):
    def __init__(self,
                 cutting=None,
                 always_apply=False,
                 p=1.0):
        super(Cut, self).__init__(always_apply, p)
        self.cutting = cutting

    def apply(self, image, **params):
        if self.cutting:
            image = image[self.cutting:-self.cutting, :, :]

        return image

    def get_transform_init_args_names(self):
        return ("size", "cutting")


def get_transforms_train(image_size_sat,
                         img_size_ground,
                         mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225],
                         ground_cutting=0):
    satellite_transforms = A.Compose([
        A.ImageCompression(quality_lower=90, quality_upper=100, p=0.5),
        A.Resize(image_size_sat[0], image_size_sat[1], interpolation=cv2.INTER_LINEAR_EXACT, p=1.0),
        A.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.15, always_apply=False, p=0.5),
        # A.OneOf(iaa_weather_list, p=1.0),
        A.OneOf([
            A.AdvancedBlur(p=1.0),
            A.Sharpen(p=1.0),
        ], p=0.3),
        A.OneOf([
            A.GridDropout(ratio=0.4, p=1.0),
            A.CoarseDropout(max_holes=25,
                            max_height=int(0.2 * image_size_sat[0]),
                            max_width=int(0.2 * image_size_sat[0]),
                            min_holes=10,
                            min_height=int(0.1 * image_size_sat[0]),
                            min_width=int(0.1 * image_size_sat[0]),
                            p=1.0),
        ], p=0.3),
        A.Normalize(mean, std),
        ToTensorV2(),
    ])

    ground_transforms = A.Compose([Cut(cutting=ground_cutting, p=1.0),
                                   A.ImageCompression(quality_lower=90, quality_upper=100, p=0.5),
                                   A.Resize(img_size_ground[0], img_size_ground[1],
                                            interpolation=cv2.INTER_LINEAR_EXACT, p=1.0),
                                   A.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.15, always_apply=False, p=0.5),
                                   # A.OneOf(iaa_weather_list, p=1.0),
                                   A.OneOf([
                                       A.AdvancedBlur(p=1.0),
                                       A.Sharpen(p=1.0),
                                   ], p=0.3),
                                   A.OneOf([
                                       A.GridDropout(ratio=0.5, p=1.0),
                                       A.CoarseDropout(max_holes=25,
                                                       max_height=int(0.2 * img_size_ground[0]),
                                                       max_width=int(0.2 * img_size_ground[0]),
                                                       min_holes=10,
                                                       min_height=int(0.1 * img_size_ground[0]),
                                                       min_width=int(0.1 * img_size_ground[0]),
                                                       p=1.0),
                                   ], p=0.3),
                                   A.Normalize(mean, std),
                                   ToTensorV2(),
                                   ])

    return satellite_transforms, ground_transforms


weather_id = 2
def get_transforms_val(image_size_sat,
                       img_size_ground,
                       mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225],
                       ground_cutting=0):
    satellite_transforms = A.Compose(
        [A.Resize(image_size_sat[0], image_size_sat[1], interpolation=cv2.INTER_LINEAR_EXACT, p=1.0),
         iaa_weather_list[weather_id],
         A.Normalize(mean, std),
         ToTensorV2(),
         ])

    ground_transforms = A.Compose([Cut(cutting=ground_cutting, p=1.0),
                                   A.Resize(img_size_ground[0], img_size_ground[1],
                                            interpolation=cv2.INTER_LINEAR_EXACT, p=1.0),
                                   iaa_weather_list[weather_id],
                                   A.Normalize(mean, std),
                                   ToTensorV2(),
                                   ])

    return satellite_transforms, ground_transforms
