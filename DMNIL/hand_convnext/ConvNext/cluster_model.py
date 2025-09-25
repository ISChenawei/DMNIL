import collections
import numpy as np
from abc import ABC
import torch
import torch.nn.functional as F
from torch import nn, autograd
from collections import defaultdict
import random
class CM(autograd.Function):

    @staticmethod
    def forward(ctx, inputs, targets, features, momentum):
        ctx.features = features
        ctx.momentum = momentum
        ctx.save_for_backward(inputs, targets)
        outputs = inputs.mm(ctx.features.t())

        return outputs

    @staticmethod
    def backward(ctx, grad_outputs):
        inputs, targets = ctx.saved_tensors
        grad_inputs = None
        if ctx.needs_input_grad[0]:
            grad_inputs = grad_outputs.mm(ctx.features)

        # momentum update
        for x, y in zip(inputs, targets):
            ctx.features[y] = ctx.momentum * ctx.features[y] + (1. - ctx.momentum) * x
            ctx.features[y] /= ctx.features[y].norm()

        return grad_inputs, None, None, None


def cm(inputs, indexes, features, momentum=0.5):
    return CM.apply(inputs, indexes, features, torch.Tensor([momentum]).to(inputs.device))


class CM_Hard(autograd.Function):

    @staticmethod
    def forward(ctx, inputs, targets, features, momentum):
        ctx.features = features
        ctx.momentum = momentum
        ctx.save_for_backward(inputs, targets)
        outputs = inputs.mm(ctx.features.t())

        return outputs

    @staticmethod
    def backward(ctx, grad_outputs):
        inputs, targets = ctx.saved_tensors
        grad_inputs = None
        if ctx.needs_input_grad[0]:
            grad_inputs = grad_outputs.mm(ctx.features)

        batch_centers = collections.defaultdict(list)
        for instance_feature, index in zip(inputs, targets.tolist()):
            batch_centers[index].append(instance_feature)

        for index, features in batch_centers.items():
            distances = []
            for feature in features:
                distance = feature.unsqueeze(0).mm(ctx.features[index].unsqueeze(0).t())[0][0]
                distances.append(distance.cpu().numpy())

            median = np.argmin(np.array(distances))
            ctx.features[index] = ctx.features[index] * ctx.momentum + (1 - ctx.momentum) * features[median]
            ctx.features[index] /= ctx.features[index].norm()

        return grad_inputs, None, None, None


def cm_hard(inputs, indexes, features, momentum=0.5):
    return CM_Hard.apply(inputs, indexes, features, torch.Tensor([momentum]).to(inputs.device))


class ClusterMemory(nn.Module, ABC):
    def __init__(self, num_features, num_samples, temp=0.05, momentum=0.2, use_hard=False):
        super(ClusterMemory, self).__init__()
        self.num_features = num_features
        self.num_samples = num_samples
        self.momentum = momentum
        self.temp = temp
        self.use_hard = use_hard

        self.register_buffer('features', torch.zeros(num_samples, num_features))

    def forward(self, inputs, targets,ca=None,training_momentum=None):

        inputs = F.normalize(inputs, dim=1).cuda()
        if training_momentum == None:
            if self.use_hard:
                outputs = cm_hard(inputs, targets, self.features, self.momentum)
            else:
                outputs = cm(inputs, targets, self.features, self.momentum)
        else:
            if self.use_hard:
                outputs = cm_hard(inputs, targets, self.features, training_momentum)
            else:
                outputs = cm(inputs, targets, self.features, training_momentum)

        outputs /= self.temp
        if ca == None:
            loss = F.cross_entropy(outputs, targets)
        else:
            loss = (F.cross_entropy(outputs, targets,reduction='none')*ca).mean()
            # return outputs

        return loss

class DHML(nn.Module):
    def __init__(self, num_features, num_samples, temp=0.05, momentum=0.2, use_hard=False, adv_weight=0.5):
        super(DHML, self).__init__()
        self.num_features = num_features
        self.num_samples = num_samples

        self.temp = temp
        self.use_hard = use_hard
        self.momentum = momentum
        self.adv_weight = adv_weight

        self.register_buffer('features', torch.zeros(num_samples, num_features))
        self.register_buffer('short_term_features', torch.zeros(num_samples, num_features))
        self.register_buffer('long_term_features', torch.zeros(num_samples, num_features))

    def forward(self, inputs, targets, ca=None, training_momentum=None):

        inputs = F.normalize(inputs, dim=1).cuda()
        if training_momentum == None:
            if self.use_hard:
                outputs = cm_hard(inputs, targets, self.features, self.momentum)
            else:
                outputs = cm(inputs, targets, self.features, self.momentum)
        else:
            if self.use_hard:
                outputs = cm_hard(inputs, targets, self.features, training_momentum)
            else:
                outputs = cm(inputs, targets, self.features, training_momentum)

        outputs /= self.temp
        if ca == None:
            loss = F.cross_entropy(outputs, targets)
        else:
            loss = (F.cross_entropy(outputs, targets, reduction='none') * ca).mean()*0
            # return outputs

        with torch.no_grad():
            self.short_term_features[targets] = (
                    0.2 * inputs + (1 - 0.3) * self.short_term_features[targets]
            )

        with torch.no_grad():
            alpha = torch.sigmoid(torch.mean(torch.norm(inputs - self.short_term_features[targets], dim=1)))
            self.long_term_features[targets] = (
                    alpha * self.short_term_features[targets] + (1 - alpha) * self.long_term_features[targets]
            )

        combined_features = 0.7 * self.long_term_features.clone() + 0.3 * self.short_term_features.clone()

        if training_momentum == None:
            if self.use_hard:
                outputs1 = cm_hard(inputs, targets, combined_features, self.momentum)
            else:
                outputs1 = cm(inputs, targets, combined_features, self.momentum)
        else:
            if self.use_hard:
                outputs1 = cm_hard(inputs, targets, combined_features, training_momentum)
            else:
                outputs1 = cm(inputs, targets, combined_features, training_momentum)
        outputs1 /= self.temp

        if ca is not None:
            classification_loss = (F.cross_entropy(outputs1, targets, reduction='none') * ca).mean()
        else:
            classification_loss = F.cross_entropy(outputs1, targets)

        return loss +classification_loss*0.2


class EM(autograd.Function):
    @staticmethod
    def forward(ctx, inputs, indexes, features, momentum):
        ctx.features = features
        ctx.momentum = momentum
        ctx.save_for_backward(inputs, indexes)
        outputs = inputs.mm(ctx.features.t())
        return outputs

    @staticmethod
    def backward(ctx, grad_outputs):
        inputs, indexes = ctx.saved_tensors
        grad_inputs = None
        if ctx.needs_input_grad[0]:
            grad_inputs = grad_outputs.mm(ctx.features)

        # momentum update, not applied for meta learning
        for x, y in zip(inputs, indexes):
            ctx.features[y] = ctx.momentum * ctx.features[y] + (1. - ctx.momentum) * x
            ctx.features[y] /= ctx.features[y].norm()

        return grad_inputs, None, None, None


def em(inputs, indexes, features, momentum=0.5):
    return EM.apply(inputs, indexes, features, torch.Tensor([momentum]).to(inputs.device))


def pairwise_distance(features_q, features_g):
    x = features_q#torch.from_numpy(features_q)
    y = features_g#torch.from_numpy(features_g)
    m, n = x.size(0), y.size(0)
    x = x.view(m, -1)
    y = y.view(n, -1)
    dist_m = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(m, n) + \
           torch.pow(y, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    dist_m.addmm_(1, -2, x, y.t())
    return dist_m#.numpy()

class ClusterMemoryV2(nn.Module):
    def __init__(self, num_features, num_samples,num_cluster, temp=0.05, momentum=0.2):
        super(ClusterMemoryV2, self).__init__()
        self.num_features = num_features
        self.num_samples = num_samples
        self.momentum = momentum
        self.temp = temp
        self.devices = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.register_buffer('features', torch.zeros(num_samples, num_features))
        # features--(source centers+tgt features)
        self.register_buffer('labels', torch.zeros(num_samples).int())
        self.register_buffer('cam', torch.zeros(num_samples).int())
        self.cam_mem = defaultdict(list)
        # labels--(each src and predicted tgt id and outliers), 13638
    def cam2uid_(self):
        uid2cam = zip(range(self.num_samples), self.cam)
        self.cam2uid = defaultdict(list)
        for uid, cam in uid2cam:
            self.cam2uid[int(cam.cpu().data)].append(uid)
        # print(self.cam2uid)
        self.allcam = torch.unique(self.cam).cpu().numpy().tolist()
        print(self.allcam)
        # self.cam_mem = defaultdict(list)
    def cam_mem_gen(self):
        num_c_total=0
        for c in self.allcam:
            self.cam_mem[c],num_c = self.generate_cluster_features(self.labels,self.features,c)
            num_c_total= num_c_total+num_c
        print(num_c_total)

    def updateEM(self, inputs, indexes):
        # momentum update
        for x, y in zip(inputs, indexes):
            self.features[y] = self.momentum * self.features[y] + (1. - self.momentum) * x
            self.features[y] = self.features[y]/self.features[y].norm()
        # del inputs,indexes

    def forward(self, inputs, indexes,cameras,neighbor_eps=0.9):
        self.thresh=-1
        self.neighbor_eps  = neighbor_eps
        inputs = F.normalize(inputs, dim=1)#.cuda()

        # print(indexes)
        sim = em(inputs, indexes, self.features, self.momentum)#B N 
        sim_exp =sim /self.temp  # 64*13638
        B = inputs.size(0)
        mask_instance = self.compute_mask(sim.size(), indexes,device=sim.device)
        sim_exp_intra = sim_exp #* mask_intra
        score_intra =   F.softmax(sim_exp_intra,dim=1)
        score_intra = score_intra.clamp_min(1e-8)
        ins_loss = -score_intra.masked_select(mask_instance.bool()).log().mean()
        return ins_loss#* 0.6

    def compute_mask(self, size, img_ids,device=None):
        mask_instance = torch.zeros(size, device=device)
        mask_instance[torch.arange(size[0]), img_ids] = 1
        return mask_instance#, mask_intra, mask_inter
    def compute_mask_camwise(self, size, img_ids, cam_ids, device):

        mask_intra = torch.ones(size, device=device)#zeros
        for i, cam in enumerate(cam_ids):
            intra_cam_ids = self.cam2uid[cam]
            mask_intra[i, intra_cam_ids] = 1
        mask_instance = torch.zeros(size, device=device)
        mask_instance[torch.arange(size[0]), img_ids] = 1
        return mask_intra,mask_instance
    def generate_cluster_features(self,labels, features,cam_id):
        centers = collections.defaultdict(list)
        for i, label in enumerate(self.labels):
            # print(int(self.cam[i]),int(cam_id))
            if (label == -1) or (int(self.cam[i]) != int(cam_id)):
                continue
            centers[int(label)].append(self.features[i])

        centers = [
            torch.stack(centers[idx], dim=0).mean(0) for idx in sorted(centers.keys())
        ]

        centers = torch.stack(centers, dim=0).cuda()
        print('cam cluster',cam_id,centers.size(0))
        return centers, centers.size(0)

    def generate_cluster_features_all(self,labels, features):
        centers = collections.defaultdict(list)
        for i, label in enumerate(labels):
            if (label == -1):
                continue
            centers[labels[i]].append(features[i])

        centers = [
            torch.stack(centers[idx], dim=0).mean(0) for idx in sorted(centers.keys())
        ]

        centers = torch.stack(centers, dim=0).cuda()
        return centers
