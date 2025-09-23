import torch.nn as nn
from .ConvNext import make_convnext_model
import torch
import numpy as np


class two_view_net(nn.Module):
    def __init__(self, class_num, block=4, return_f=False, resnet=False):
        super(two_view_net, self).__init__()
        self.model_1 = make_convnext_model(num_class=class_num, block=block, return_f=return_f, resnet=resnet)

        # 1. temperature factor for contrastive learning
        # self.logit_scale = torch.nn.Parameter(torch.ones([]) * np.log(1 / 0.02))
        # self.logit_scale = torch.scalar_tensor(3.569)
        self.logit_scale = torch.nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.logit_scale_blocks = torch.nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        # 2. weight for blocks_infoNCE
        self.w_blocks1 = torch.nn.Parameter(torch.ones([]))
        self.w_blocks2 = torch.nn.Parameter(torch.ones([]))
        self.w_blocks3 = torch.nn.Parameter(torch.ones([]))


    def get_config(self):
        input_size = (3, 224, 224)
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)

        config = {
            'input_size': input_size,
            'mean': mean,
            'std': std
        }
        return config

    def forward(self, x1, x2=None,):
        # if x1 is None:
        #     y1 = None
        # else:
        #     y1 = self.model_1(x1)
        #     # print("pause")
        #
        # if x2 is None:
        #     y2 = None
        # else:
        #     y2 = self.model_1(x2)
        # return y1, y2

        if x2 is not None:
            y1 = self.model_1(x1)
            y2 = self.model_1(x2)
            return y1, y2
        else:
            y1 = self.model_1(x1)
            return y1


class three_view_net(nn.Module):
    def __init__(self, class_num, share_weight=False, block=4, return_f=False, resnet=False):
        super(three_view_net, self).__init__()
        self.share_weight = share_weight
        self.model_1 = make_convnext_model(num_class=class_num, block=block, return_f=return_f, resnet=resnet)

        if self.share_weight:
            self.model_2 = self.model_1
        else:
            self.model_2 = make_convnext_model(num_class=class_num, block=block, return_f=return_f, resnet=resnet)

    def forward(self, x1, x2, x3, x4=None):  # x4 is extra data
        if x1 is None:
            y1 = None
        else:
            y1 = self.model_1(x1)

        if x2 is None:
            y2 = None
        else:
            y2 = self.model_2(x2)

        if x3 is None:
            y3 = None
        else:
            y3 = self.model_1(x3)

        if x4 is None:
            return y1, y2, y3
        else:
            y4 = self.model_2(x4)
        return y1, y2, y3, y4


def make_model(opt):
    if opt.views == 2:
        model = two_view_net(opt.nclasses, block=opt.block, return_f=opt.triplet_loss, resnet=opt.resnet)
    # elif opt.views == 3:
    #     model = three_view_net(opt.nclasses, share_weight=opt.share, block=opt.block, return_f=opt.triplet_loss,
    #                            resnet=opt.resnet)
    return model
