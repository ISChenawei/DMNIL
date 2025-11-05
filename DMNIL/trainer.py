import time
import torch
from tqdm import tqdm
from .utils import AverageMeter
from torch.cuda.amp import autocast
import torch.nn.functional as F
import torch.nn as nn
from thop import profile
def predict(train_config, model, dataloader):
    model.eval()
    draw_vis = False
    if draw_vis:
        import cv2
        import os
        import numpy as np
        pic_path = "./draw_vis"
        iterations = int(len(os.listdir(pic_path)) / 2)
        for i in range(iterations):
            uav_ori = cv2.imread(rf"{pic_path}/{i}_uav.jpg")
            sat_ori = cv2.imread(rf"{pic_path}/{i}_sat.jpg")

            uav_shape = uav_ori.shape[:-1]
            sat_shape = sat_ori.shape[:-1]

            uav = cv2.resize(uav_ori, (384, 384), interpolation=cv2.INTER_LINEAR).astype('float32') / 255.0
            sat = cv2.resize(sat_ori, (384, 384), interpolation=cv2.INTER_LINEAR).astype('float32') / 255.0

            uav = torch.tensor(uav).permute(2, 0, 1)
            sat = torch.tensor(sat).permute(2, 0, 1)

            # 图像标准化
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
            import torchvision.transforms as transforms
            normalize = transforms.Normalize(mean=mean, std=std)
            uav = normalize(uav)[None, :, :, :]
            sat = normalize(sat)[None, :, :, :]

            with torch.no_grad():
                with autocast():
                    uav = uav.to(train_config.device)
                    sat = sat.to(train_config.device)

                    img_feature_uav = model(uav)[1]
                    img_feature_uav = F.normalize(img_feature_uav, dim=1)

                    img_feature_sat = model(sat)[1]
                    img_feature_sat = F.normalize(img_feature_sat, dim=1)

                    heat_map_uav = img_feature_uav[0].permute(1, 2, 0)
                    heat_map_uav = torch.mean(heat_map_uav, dim=2).detach().cpu().numpy()
                    heat_map_uav = (heat_map_uav - heat_map_uav.min()) / (heat_map_uav.max() - heat_map_uav.min())
                    heat_map_uav = cv2.resize(heat_map_uav, [uav_shape[1], uav_shape[0]])

                    heat_map_sat = img_feature_sat[0].permute(1, 2, 0)
                    heat_map_sat = torch.mean(heat_map_sat, dim=2).detach().cpu().numpy()
                    heat_map_sat = (heat_map_sat - heat_map_sat.min()) / (heat_map_sat.max() - heat_map_sat.min())
                    heat_map_sat = cv2.resize(heat_map_sat, [sat_shape[1], sat_shape[0]])

                    #  colorize
                    colored_image_uav = cv2.applyColorMap((heat_map_uav * 255).astype(np.uint8), cv2.COLORMAP_JET)
                    colored_image_sat = cv2.applyColorMap((heat_map_sat * 255).astype(np.uint8), cv2.COLORMAP_JET)

                    # 设置半透明度（alpha值）
                    alpha = 0.5
                    # 将两个图像进行叠加
                    blended_image_uav = cv2.addWeighted(uav_ori, alpha, colored_image_uav, 1 - alpha, 0)
                    blended_image_sat = cv2.addWeighted(sat_ori, alpha, colored_image_sat, 1 - alpha, 0)


                    out_path = "/SUES-D-S-Rank"
                    cv2.imwrite(rf"{out_path}/{i}_uav_vis.jpg", blended_image_uav)
                    cv2.imwrite(rf"{out_path}/{i}_sat_vis.jpg", blended_image_sat)

        return 0

    if train_config.verbose:
        bar = tqdm(dataloader, total=len(dataloader))
    else:
        bar = dataloader

    img_features_list = []
    path_list = []
    ids_list = []
    with torch.no_grad():

        for img, ids,img_paths in bar:
            start_time = time.time()
            ids_list.append(ids)
            path_list.extend(img_paths)
            with autocast():
                img = img.to(train_config.device)

                if train_config.handcraft_model is not True:
                    img_feature = model(img)
                else:
                    img_feature = model(img)[-2]

                # normalize is calculated in fp32
                if train_config.normalize_features:
                    img_feature = F.normalize(img_feature, dim=-1)

            # save features in fp32 for sim calculation
            img_features_list.append(img_feature.to(torch.float32))
        # end_time = time.time()
        # elapsed_time = end_time -start_time
        # totle_time += elapsed_time
        # num_batches+=1
        # keep Features on GPU
        img_features = torch.cat(img_features_list, dim=0)
        ids_list = torch.cat(ids_list, dim=0).to(train_config.device)
    # avg_fps = num_batches/totle_time if totle_time>0 else 0
    # print(f"Average FPS:{avg_fps:.2f}")
    if train_config.verbose:
        bar.close()

    return img_features, ids_list,path_list

