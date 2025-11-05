from __future__ import print_function, absolute_import
import time
import collections
from collections import OrderedDict
import numpy as np
import torch
import random
import copy
from torch.cuda.amp import autocast
from tqdm import tqdm
from .Utils.meters import AverageMeter
from .Utils.rerank import re_ranking
import torch.nn.functional as F
def fliplr(img):
    '''flip horizontal'''
    inv_idx = torch.arange(img.size(3)-1,-1,-1).long()  # N x C x H x W
    img_flip = img.index_select(3,inv_idx)
    return img_flip

def extract_cnn_feature(model, inputs):
    # inputs = to_torch(inputs).cuda()
    inputs = inputs.cuda()
    # inputs1 = inputs
    # print(inputs)
    output1, output2 = model(inputs, inputs)
    outputs, outputs_s = output1[0], output2[0]  # -- for contrastive
    outputs = outputs.data.cpu()
    outputs_s = outputs_s.data.cpu()
    return outputs, outputs_s

def extract_features(model, data_loader, print_freq=50,flip=True):
    model.eval()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    features = OrderedDict()
    features_s = OrderedDict()
    labels = OrderedDict()

    end = time.time()
    with torch.no_grad():
        bar = tqdm(enumerate(data_loader),total=len(data_loader))
        with (autocast()):
            for i, (imgs, fnames, pids, _, _) in bar:
                data_time.update(time.time() - end)

                outputs,outputs_s = extract_cnn_feature(model, imgs)
                flip = fliplr(imgs)
            # print(flip)
                outputs_flip,outputs_flip_s = extract_cnn_feature(model, flip)

                for fname, output,output_flip,pid in zip(fnames, outputs,outputs_flip, pids):
                    features[fname] =  (output.detach() + output_flip.detach())/2.0#F.normalize((output.detach() + output_flip.detach())/2.0, dim=-1)
                    labels[fname] = pid

                for fname, output,output_flip,pid in zip(fnames, outputs_s,outputs_flip_s, pids):
                    features_s[fname] =  (output.detach() + output_flip.detach())/2.0 #F.normalize((output.detach() + output_flip.detach())/2.0, dim=-1) #
                    labels[fname] = pid

                batch_time.update(time.time() - end)
                end = time.time()

                if (i + 1) % print_freq == 0:
                    print('Extract Features: [{}/{}]\t'
                        'Time {:.3f} ({:.3f})\t'
                        'Data {:.3f} ({:.3f})\t'
                        .format(i + 1, len(data_loader),
                                batch_time.val, batch_time.avg,
                                data_time.val, data_time.avg))
        print("total time: {}".format(time.time()-end))
        return features,features_s#, labels
