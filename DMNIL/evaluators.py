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
from .evaluation_metrics import cmc,mean_ap
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


def pairwise_distance(features, query=None, gallery=None):
    if query is None and gallery is None:
        n = len(features)
        x = torch.cat(list(features.values()))
        x = x.view(n, -1)
        dist_m = torch.pow(x, 2).sum(dim=1, keepdim=True) * 2
        dist_m = dist_m.expand(n, n) - 2 * torch.mm(x, x.t())
        return dist_m

    x = torch.cat([features[f].unsqueeze(0) for f, _, _ in query], 0)
    y = torch.cat([features[f].unsqueeze(0) for f, _, _ in gallery], 0)
    m, n = x.size(0), y.size(0)
    x = x.view(m, -1)
    y = y.view(n, -1)
    dist_m = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(m, n) + \
           torch.pow(y, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    dist_m.addmm_(1, -2, x, y.t())
    return dist_m, x.numpy(), y.numpy()


def evaluate_all(query_features, gallery_features, distmat, query=None, gallery=None,
                 query_ids=None, gallery_ids=None,
                 query_cams=None, gallery_cams=None,
                 cmc_topk=(1, 5, 10), cmc_flag=False,regdb=False):
    if query is not None and gallery is not None:
        query_ids = [pid for _, pid, _ in query]
        gallery_ids = [pid for _, pid, _ in gallery]
        query_cams = [cam for _, _, cam in query]
        gallery_cams = [cam for _, _, cam in gallery]
    else:
        assert (query_ids is not None and gallery_ids is not None
                and query_cams is not None and gallery_cams is not None)

    # Compute mean AP
    mAP = mean_ap(distmat, query_ids, gallery_ids, query_cams, gallery_cams,regdb=regdb)
    print('Mean AP: {:4.1%}'.format(mAP))

    if (not cmc_flag):
        return mAP

    cmc_configs = {
        'market1501': dict(separate_camera_set=False,
                           single_gallery_shot=False,
                           first_match_break=True),}
    cmc_scores = {name: cmc(distmat, query_ids, gallery_ids,
                            query_cams, gallery_cams,regdb=regdb, **params)
                  for name, params in cmc_configs.items()}

    print('CMC Scores:')
    for k in cmc_topk:
        print('  top-{:<4}{:12.1%}'.format(k, cmc_scores['market1501'][k-1]))
    return cmc_scores['market1501'], mAP


class Evaluator(object):
    def __init__(self, model):
        super(Evaluator, self).__init__()
        self.model = model

    def evaluate(self, data_loader, query, gallery, cmc_flag=False, rerank=False,modal=0,regdb=False):
        features, _ = extract_features(self.model, data_loader,mode=modal)
        # print(features,features) 
        distmat, query_features, gallery_features = pairwise_distance(features, query, gallery)
        # print(distmat)
        results = evaluate_all(query_features, gallery_features, distmat, query=query, gallery=gallery, cmc_flag=cmc_flag,regdb=regdb)

        if (not rerank):
            return results

        print('Applying person re-ranking ...') 
        distmat_qq, _, _ = pairwise_distance(features, query, query)
        distmat_gg, _, _ = pairwise_distance(features, gallery, gallery)
        distmat = re_ranking(distmat.numpy(), distmat_qq.numpy(), distmat_gg.numpy())
        return evaluate_all(query_features, gallery_features, distmat, query=query, gallery=gallery, cmc_flag=cmc_flag)
