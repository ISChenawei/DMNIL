from __future__ import print_function, absolute_import
import time

from .Utils.faiss_rerank import compute_ranked_list_cm
from .Utils.meters import AverageMeter
import torch.nn as nn
import torch
from torch.nn import functional as F
from torch.nn import Module
import collections
from torch import einsum
from torch.autograd import Variable
import numpy as np
def pdist_torch(emb1, emb2):
    '''
    compute the eucilidean distance matrix between embeddings1 and embeddings2
    using gpu
    '''
    m, n = emb1.shape[0], emb2.shape[0]
    emb1_pow = torch.pow(emb1, 2).sum(dim=1, keepdim=True).expand(m, n)
    emb2_pow = torch.pow(emb2, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    dist_mtx = emb1_pow + emb2_pow
    dist_mtx = dist_mtx.addmm_(1, -2, emb1, emb2.t())
    # dist_mtx = dist_mtx.clamp(min = 1e-12)
    dist_mtx = dist_mtx.clamp(min=1e-12).sqrt()
    return dist_mtx


def softmax_weights(dist, mask):
    max_v = torch.max(dist * mask, dim=1, keepdim=True)[0]
    diff = dist - max_v
    Z = torch.sum(torch.exp(diff) * mask, dim=1, keepdim=True) + 1e-6  # avoid division by zero
    W = torch.exp(diff) * mask / Z
    return W
def normalize(x, axis=-1):
    x = 1. * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
    return x
class TripletLoss_WRT(nn.Module):
    """Weighted Regularized Triplet'."""

    def __init__(self):
        super(TripletLoss_WRT, self).__init__()
        self.ranking_loss = nn.SoftMarginLoss()

    def forward(self, inputs, targets, normalize_feature=True):
        if normalize_feature:
            inputs = normalize(inputs, axis=-1)
        dist_mat = pdist_torch(inputs, inputs)

        N = dist_mat.size(0)
        is_pos = targets.expand(N, N).eq(targets.expand(N, N).t()).float()
        is_neg = targets.expand(N, N).ne(targets.expand(N, N).t()).float()

        dist_ap = dist_mat * is_pos
        dist_an = dist_mat * is_neg

        weights_ap = softmax_weights(dist_ap, is_pos)
        weights_an = softmax_weights(-dist_an, is_neg)
        furthest_positive = torch.sum(dist_ap * weights_ap, dim=1)
        closest_negative  = torch.sum(dist_an * weights_an, dim=1)

        y = furthest_positive.new().resize_as_(furthest_positive).fill_(1)
        loss = self.ranking_loss(closest_negative - furthest_positive, y)

        return loss#, correct


class KLDivLoss(nn.Module):
    def __init__(self):
        super(KLDivLoss, self).__init__()
    def forward(self, pred, label):
        # pred: 2D matrix (batch_size, num_classes)
        # label: 1D vector indicating class number
        T=3

        predict = F.log_softmax(pred/T,dim=1)
        target_data = F.softmax(label/T,dim=1)
        target_data =target_data+10**(-7)
        target = Variable(target_data.data.cuda(),requires_grad=False)
        loss=T*T*((target*(target.log()-predict)).sum(1).sum()/target.size()[0])
        return loss

def compute_cross_agreement_dd(features_g, features_p,features_g_s,features_p_s, k=20, search_option=3):
    print("Compute cross agreement score...")
    N, D = features_p.size()
    M, D = features_g.size()
    score = torch.FloatTensor()
    end = time.time()
    ranked_list_g = compute_ranked_list_cm(features_g,features_p, k=k, search_option=search_option, verbose=False)
    ranked_list_p_i = compute_ranked_list_cm(features_p_s,features_p_s, k=k, search_option=search_option, verbose=False)
    score_all =[]
    for i in range(M):
        intersect_i = torch.FloatTensor(
            [len(np.intersect1d(ranked_list_g[i], ranked_list_p_i[j])) for j in range(N)])
        union_i = torch.FloatTensor(
            [len(np.union1d(ranked_list_g[i], ranked_list_p_i[j])) for j in range(N)])
        score_i = intersect_i / union_i
        score_all.append(score_i)
    score = torch.cat(score_all, dim=0)
    return score_i


class Trainer_DMNIL(object):
    def __init__(self, encoder, memory=None, matcher_sat=None, matcher_dro=None):
        super(Trainer_DMNIL, self).__init__()
        self.encoder = encoder
        self.memory_dro = memory
        self.memory_sat = memory
        self.wise_memory_dro = memory
        self.wise_memory_sat = memory
        self.nameMap_dro = []
        self.nameMap_sat = []
        self.criterion_kl = KLDivLoss()
        self.cmlabel = 0
        self.memory_dro_s = memory
        self.memory_sat_s = memory
        self.wise_memory_dro_s = memory
        self.wise_memory_sat_s = memory
        self.shared_memory = memory
        self.shared_memory_s = memory
        self.htsd = 0

        self.hm = 0
        self.ht = 0

    def train(self, epoch, data_loader_dro, data_loader_sat, optimizer, print_freq=10, train_iters=400):
        self.encoder.train()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        loss_dro_log = AverageMeter()
        loss_sat_log = AverageMeter()
        dro_sat_loss_log = AverageMeter()
        sat_dro_loss_log = AverageMeter()
        sat_sat_loss_log = AverageMeter()
        dro_dro_loss_log = AverageMeter()

        lamda_s_neibor = 0.5
        lamda_d_neibor = 1
        lamda_sd = 1
        lamda_c = 0.1  # 0.1

        end = time.time()
        for i in range(train_iters):

            # load data
            inputs_dro = data_loader_dro.next()
            inputs_sat = data_loader_sat.next()
            data_time.update(time.time() - end)

            inputs_dro, labels_dro, indexes_dro, cids_dro, name_dro = self._parse_data_dro(inputs_dro)  # inputs_dro1

            inputs_sat, inputs_sat1, labels_sat, indexes_sat, cids_sat, name_sat = self._parse_data_sat(inputs_sat)
            # forward
            inputs_sat = torch.cat((inputs_sat, inputs_sat1), 0)
            labels_sat = torch.cat((labels_sat, labels_sat), -1)
            cids_sat = torch.cat((cids_sat, cids_sat), -1)

            indexes_dro = torch.tensor([self.nameMap_dro[name] for name in name_dro]).cuda()
            indexes_sat = torch.tensor([self.nameMap_sat[name] for name in name_sat])
            indexes_sat = torch.cat((indexes_sat, indexes_sat), -1).cuda()
            outputs, outputs_s = self._forward(inputs_sat, inputs_dro)
            f_out_sat = outputs[0]
            f_out_dro = outputs_s[0]
            f_out_sat_s = outputs[0]
            f_out_dro_s = outputs_s[0]
            index_sat = indexes_sat
            index_dro =indexes_dro
            # f_out_sat, f_out_dro, f_out_sat_s, f_out_dro_s, labels_sat, labels_dro, \
            #     cid_sat, cid_dro, index_sat, index_dro = self._forward(inputs_sat, inputs_dro)

            loss_dro_s = torch.tensor([0.]).cuda()
            loss_sat_s = torch.tensor([0.]).cuda()
            loss_dro = torch.tensor([0.]).cuda()
            loss_sat = torch.tensor([0.]).cuda()
            dro_sat_loss = torch.tensor([0.]).cuda()
            sat_dro_loss = torch.tensor([0.]).cuda()
            sat_sat_loss = torch.tensor([0.]).cuda()
            dro_dro_loss = torch.tensor([0.]).cuda()

            dro_sat_loss_s = torch.tensor([0.]).cuda()
            sat_dro_loss_s = torch.tensor([0.]).cuda()
            sat_sat_loss_s = torch.tensor([0.]).cuda()
            dro_dro_loss_s = torch.tensor([0.]).cuda()
            loss_shared = torch.tensor([0.]).cuda()
            loss_shared_s = torch.tensor([0.]).cuda()

            loss_dro_s = lamda_sd * self.memory_dro_s(f_out_dro_s, labels_dro)
            loss_sat_s = lamda_sd * self.memory_sat_s(f_out_sat_s, labels_sat)
            loss_dro = self.memory_dro(f_out_dro, labels_dro)
            loss_sat = self.memory_sat(f_out_sat, labels_sat)

            thresh = 0.9
            hm_thresh = 0.9

            if epoch >= self.hm:  # self.cmlabel:
                if epoch >= self.ht:  # self.cmlabel:
                    if epoch % 2 == 0:
                        with torch.no_grad():
                            sim_prob_sat_dro = self.wise_memory_sat_s.features.detach()[index_sat].mm(self.wise_memory_dro_s.features.detach().data.t())
                            sim_sat_dro = self.wise_memory_sat.features.detach()[index_sat].mm(self.wise_memory_dro.features.detach().data.t())
                            k = 10 
                            k1 = 20
                            _, indices = sim_sat_dro.topk(k, dim=1)
                            _, indices1 = sim_sat_dro.topk(k1, dim=1)

                            nearest_sat_dro = sim_sat_dro.max(dim=1, keepdim=True)[0]
                            nearest_prob_sat_dro = sim_prob_sat_dro.max(dim=1, keepdim=True)[0]
                            mask_neighbor_sat_dro = torch.gt(sim_sat_dro, nearest_sat_dro * thresh).detach().data
                            mask_neighbor_prob_sat_dro = torch.gt(sim_prob_sat_dro,nearest_prob_sat_dro * thresh)
                            num_neighbor_sat_dro = mask_neighbor_sat_dro.mul(mask_neighbor_prob_sat_dro).sum(dim=1) + 1
                        sim_sat_dro = F.normalize(f_out_sat, dim=1).mm(self.wise_memory_dro.features.detach().data.t())
                        sim_prob_sat_dro = F.normalize(f_out_sat_s, dim=1).mm(self.wise_memory_dro_s.features.detach().data.t())
                        sim_sat_dro_exp = sim_sat_dro / 0.05
                        score_intra_sat_dro = F.softmax(sim_sat_dro_exp, dim=1)
                        score_intra_sat_dro = score_intra_sat_dro.clamp_min(1e-8)
                        sat_dro_loss = -score_intra_sat_dro.log().mul(mask_neighbor_sat_dro).mul(mask_neighbor_prob_sat_dro).sum( dim=1)
                        sat_dro_loss = 0.5 * lamda_d_neibor * sat_dro_loss.div(num_neighbor_sat_dro).mean()
                        neighbor_sim = sim_sat_dro[:, indices]
                        p = F.softmax(neighbor_sim, dim=1)
                        q = torch.ones_like(p) / p.size(1) 
                        epsilon = 1e-8
                        p = (p + epsilon) / (1 + epsilon * p.size(1)) 
                        consistency_loss = F.kl_div(p.log(), q, reduction='batchmean')
                        sat_dro_loss += 0.01 * consistency_loss

                        sim_prob_sat_dro_exp = sim_prob_sat_dro / 0.05  # 64*13638
                        score_intra_sat_dro_s = F.softmax(sim_prob_sat_dro_exp, dim=1)
                        score_intra_sat_dro_s = score_intra_sat_dro_s.clamp_min(1e-8)

                        sat_dro_loss_s = -score_intra_sat_dro_s.log().mul(mask_neighbor_sat_dro).mul(
                            mask_neighbor_prob_sat_dro).sum(dim=1)

                        sat_dro_loss_s = 0.1 * lamda_s_neibor * sat_dro_loss_s.div(num_neighbor_sat_dro).mean()
                        neighbor_sim = sim_sat_dro[:, indices1]
                        p = F.softmax(neighbor_sim, dim=1)
                        p = (p + epsilon) / (1 + epsilon * p.size(1))
                        mutual_info = (p * p.log()).sum(dim=1).mean().clamp_min(0)
                        sat_dro_loss_s += 0.1 * mutual_info

                    else:
                        with torch.no_grad():
                            sim_prob_sat_dro = self.wise_memory_dro_s.features.detach()[index_dro].mm(
                                self.wise_memory_sat_s.features.detach().data.t())  # F.softmax(F.normalize(f_out_sat_s, dim=1).mm(self.wise_memory_dro_s.features.detach().data.t())/0.05,dim=1)#B N
                            sim_sat_dro = self.wise_memory_dro.features.detach()[index_dro].mm(
                                self.wise_memory_sat.features.detach().data.t())  # F.softmax(F.normalize(f_out_sat, dim=1).mm(self.wise_memory_dro.features.detach().data.t())/0.05,dim=1)
                            k = 10  
                            k1 = 20
                            _, indices = sim_sat_dro.topk(k, dim=1)
                            _, indices1 = sim_sat_dro.topk(k1, dim=1)
                            nearest_sat_dro = sim_sat_dro.max(dim=1, keepdim=True)[0]
                            nearest_prob_sat_dro = sim_prob_sat_dro.max(dim=1, keepdim=True)[0]
                            mask_neighbor_sat_dro = torch.gt(sim_sat_dro,
                                                             nearest_sat_dro * thresh).detach().data  # nearest_intra * self.neighbor_eps)self.neighbor_eps
                            mask_neighbor_prob_sat_dro = torch.gt(sim_prob_sat_dro,
                                                                  nearest_prob_sat_dro * thresh)  # .cuda()#nearest_intra * self.neighbor_eps)self.neighbor_eps
                            num_neighbor_sat_dro = mask_neighbor_sat_dro.mul(mask_neighbor_prob_sat_dro).sum(dim=1) + 1

                        sim_prob_sat_dro = F.normalize(f_out_dro_s, dim=1).mm(
                            self.wise_memory_sat_s.features.detach().data.t())  # F.softmax(F.normalize(f_out_sat_s, dim=1).mm(self.wise_memory_dro_s.features.detach().data.t())/0.05,dim=1)#B N
                        sim_sat_dro = F.normalize(f_out_dro, dim=1).mm(
                            self.wise_memory_sat.features.detach().data.t())  # F.softmax(F.normalize(f_out_sat, dim=1).mm(self.wise_memory_dro.features.detach().data.t())/0.05,dim=1)
                        sim_sat_dro_exp = sim_sat_dro / 0.05  # 64*13638
                        score_intra_sat_dro = F.softmax(sim_sat_dro_exp,
                                                        dim=1)  ##sim_exp_intra / sim_exp_intra.sum(dim=1, keepdim=True)#
                        score_intra_sat_dro = score_intra_sat_dro.clamp_min(1e-8)
                        dro_sat_loss = -score_intra_sat_dro.log().mul(mask_neighbor_sat_dro).mul(mask_neighbor_prob_sat_dro).sum(dim=1)
                        dro_sat_loss = lamda_d_neibor * dro_sat_loss.div(num_neighbor_sat_dro).mean()

                        sim_prob_sat_dro_exp = sim_prob_sat_dro / 0.05  # 64*13638
                        score_intra_sat_dro_s = F.softmax(sim_prob_sat_dro_exp, dim=1)
                        score_intra_sat_dro_s = score_intra_sat_dro_s.clamp_min(1e-8)
                        neighbor_sim = sim_sat_dro[:, indices1]
                        p = F.softmax(neighbor_sim, dim=1)
                        q = torch.ones_like(p) / p.size(1) 
                        epsilon = 1e-8
                        p = (p + epsilon) / (1 + epsilon * p.size(1))  
                        consistency_loss = F.kl_div(p.log(), q, reduction='batchmean')
                        dro_sat_loss += 0.01 * consistency_loss
                        dro_sat_loss = dro_sat_loss*0
                        dro_sat_loss_s = -score_intra_sat_dro_s.log().mul(mask_neighbor_sat_dro).mul(mask_neighbor_prob_sat_dro).sum(dim=1)
                        dro_sat_loss_s = lamda_s_neibor * dro_sat_loss_s.div(num_neighbor_sat_dro).mean()  # .mul(dro_sat_ca).mul(sat_dro_ca) ##
                        neighbor_sim = sim_sat_dro[:, indices]
                        p = F.softmax(neighbor_sim, dim=1)
                        p = (p + epsilon) / (1 + epsilon * p.size(1))
                        mutual_info = (p * p.log()).sum(dim=1).mean().clamp_min(0)
                        dro_sat_loss_s += 0.1 * mutual_info

                with torch.no_grad():
                    sim_prob_sat_sat = self.wise_memory_sat_s.features.detach()[index_sat].mm(
                        self.wise_memory_sat_s.features.detach().data.t())  # F.softmax(F.normalize(f_out_sat_s, dim=1).mm(self.wise_memory_sat_s.features.detach().data.t())/0.05,dim=1)#B N
                    sim_sat_sat = self.wise_memory_sat.features.detach()[index_sat].mm(
                        self.wise_memory_sat.features.detach().data.t())  # F.softmax(F.normalize(f_out_sat, dim=1).mm(self.wise_memory_sat.features.detach().data.t())/0.05,dim=1)
                    k = 10
                    k1 = 20
                    _, indices = sim_sat_sat.topk(k, dim=1)
                    _, indices1 = sim_sat_sat.topk(k1, dim=1)
                    nearest_sat_sat = sim_sat_sat.max(dim=1, keepdim=True)[0]
                    nearest_prob_sat_sat = sim_prob_sat_sat.max(dim=1, keepdim=True)[0]
                    mask_neighbor_sat_sat = torch.gt(sim_sat_sat,
                                                     nearest_sat_sat * hm_thresh).detach().data  # nearest_intra * self.neighbor_eps)self.neighbor_eps
                    mask_neighbor_prob_sat_sat = torch.gt(sim_prob_sat_sat,
                                                          nearest_prob_sat_sat * hm_thresh)  # .cuda()#nearest_intra * self.neighbor_eps)self.neighbor_eps
                    num_neighbor_sat_sat = mask_neighbor_sat_sat.mul(mask_neighbor_prob_sat_sat).sum(dim=1) + 1

                    # print('num_neighbor_sat_sat',num_neighbor_sat_sat)

                sim_prob_sat_sat = F.normalize(f_out_sat_s, dim=1).mm(
                    self.wise_memory_sat_s.features.detach().data.t())  # F.softmax(F.normalize(f_out_sat_s, dim=1).mm(self.wise_memory_sat_s.features.detach().data.t())/0.05,dim=1)#B N
                sim_sat_sat = F.normalize(f_out_sat, dim=1).mm(
                    self.wise_memory_sat.features.detach().data.t())  # F.softmax(F.normalize(f_out_sat, dim=1).mm(self.wise_memory_sat.features.detach().data.t())/0.05,dim=1)
                sim_sat_sat_exp = sim_sat_sat / 0.05  # 64*13638
                score_intra_sat_sat = F.softmax(sim_sat_sat_exp,
                                                dim=1)  ##sim_exp_intra / sim_exp_intra.sum(dim=1, keepdim=True)#
                # print('score_intra',score_intra)
                score_intra_sat_sat = score_intra_sat_sat.clamp_min(1e-8)
                # count_sat_dro = (mask_neighbor_sat_dro).sum(dim=1)
                sat_sat_loss = -score_intra_sat_sat.log().mul(mask_neighbor_sat_sat).mul(
                    mask_neighbor_prob_sat_sat).sum(
                    dim=1)  # .mul(sim_wise) mul(mask_neighbor_intra) .mul(mask_neighbor_intra)
                sat_sat_loss = lamda_d_neibor * sat_sat_loss.div(
                    num_neighbor_sat_sat).mean()  # .mul(sat_ca).mul(sat_ca).mul(sat_ca)..mul(sat_ca)mul(mask_neighbor_intra_soft) ##

                neighbor_sim = sim_sat_sat[:, indices1]
                p = F.softmax(neighbor_sim, dim=1)
                q = torch.ones_like(p) / p.size(1)
                epsilon = 1e-8
                p = (p + epsilon) / (1 + epsilon * p.size(1))  
                consistency_loss = F.kl_div(p.log(), q, reduction='batchmean')
                # print('consistency_loss', consistency_loss)
                sat_sat_loss += 0.01 * consistency_loss
                sat_sat_loss=sat_sat_loss*0
                sim_prob_sat_sat_exp = sim_prob_sat_sat / 0.05  # 64*13638
                score_intra_sat_sat_s = F.softmax(sim_prob_sat_sat_exp, dim=1)
                score_intra_sat_sat_s = score_intra_sat_sat_s.clamp_min(1e-8)
                # count_sat_dro = (mask_neighbor_sat_dro).sum(dim=1)
                sat_sat_loss_s = -score_intra_sat_sat_s.log().mul(mask_neighbor_sat_sat).mul(
                    mask_neighbor_prob_sat_sat).sum(
                    dim=1)  # .mul(sim_wise) mul(mask_neighbor_intra) .mul(mask_neighbor_intra)
                sat_sat_loss_s = lamda_s_neibor * sat_sat_loss_s.div(
                    num_neighbor_sat_sat).mean()  # .mul(sat_ca).mul(sat_ca).mul(mask_neighbor_intra_soft) ##
                neighbor_sim = sim_sat_sat[:, indices]
                p = F.softmax(neighbor_sim, dim=1) 
                p = (p + epsilon) / (1 + epsilon * p.size(1))
                # 计算互信息
                mutual_info = (p * p.log()).sum(dim=1).mean().clamp_min(0)
                sat_sat_loss_s += 0.1 * mutual_info
                # print('sat_sat_loss sat_sat_loss_s',sat_sat_loss.size(),sat_sat_loss_s.size())
                sat_sat_loss_s=sat_sat_loss_s
                # #################dro-dro

                with torch.no_grad():
                    sim_prob_dro_dro = self.wise_memory_dro_s.features.detach()[index_dro].mm(
                        self.wise_memory_dro_s.features.detach().data.t())  # F.softmax(F.normalize(f_out_dro_s, dim=1).mm(self.wise_memory_dro_s.features.detach().data.t())/0.05,dim=1)#B N
                    sim_dro_dro = self.wise_memory_dro.features.detach()[index_dro].mm(
                        self.wise_memory_dro.features.detach().data.t())  # F.softmax(F.normalize(f_out_dro, dim=1).mm(self.wise_memory_dro.features.detach().data.t())/0.05,dim=1)
                    k = 10
                    k1 = 20
                    # _, indices = sim_sat_sat.topk(k, dim=1)
                    # _, indices1 = sim_sat_sat.topk(k1, dim=1)
                    _, indices = sim_sat_sat.topk(k, dim=1)
                    _, indices1 = sim_sat_sat.topk(k1, dim=1)
                    nearest_dro_dro = sim_dro_dro.max(dim=1, keepdim=True)[0]
                    nearest_prob_dro_dro = sim_prob_dro_dro.max(dim=1, keepdim=True)[0]
                    mask_neighbor_prob_dro_dro = torch.gt(sim_prob_dro_dro,
                                                        nearest_prob_dro_dro * hm_thresh)  # .cuda()#nearest_intra * self.neighbor_eps)self.neighbor_eps
                    mask_neighbor_dro_dro = torch.gt(sim_dro_dro,
                                                   nearest_dro_dro * hm_thresh).detach().data  # nearest_intra * self.neighbor_eps)self.neighbor_eps
                    num_neighbor_dro_dro = mask_neighbor_dro_dro.mul(mask_neighbor_prob_dro_dro).sum(
                        dim=1) + 1  # .mul(sim_wise).
                    # print('num_neighbor_dro_dro',num_neighbor_dro_dro)

                sim_prob_dro_dro = F.normalize(f_out_dro_s, dim=1).mm(
                    self.wise_memory_dro_s.features.detach().data.t())  # F.softmax(F.normalize(f_out_dro_s, dim=1).mm(self.wise_memory_dro_s.features.detach().data.t())/0.05,dim=1)#B N
                sim_dro_dro = F.normalize(f_out_dro, dim=1).mm(
                    self.wise_memory_dro.features.detach().data.t())  # F.softmax(F.normalize(f_out_dro, dim=1).mm(self.wise_memory_dro.features.detach().data.t())/0.05,dim=1)
                sim_dro_dro_exp = sim_dro_dro / 0.05  # 64*13638
                score_intra_dro_dro = F.softmax(sim_dro_dro_exp,
                                              dim=1)  ##sim_exp_intra / sim_exp_intra.sum(dim=1, keepdim=True)#
                # print('score_intra',score_intra)
                score_intra_dro_dro = score_intra_dro_dro.clamp_min(1e-8)
                # count_dro_sat = (mask_neighbor_dro_sat).sum(dim=1)
                dro_dro_loss = -score_intra_dro_dro.log().mul(mask_neighbor_dro_dro).mul(mask_neighbor_prob_dro_dro).sum(
                    dim=1)  # .mul(sim_wise) mul(mask_neighbor_intra) .mul(mask_neighbor_intra)
                dro_dro_loss = lamda_d_neibor * dro_dro_loss.div(
                    num_neighbor_dro_dro).mean()  # .mul(dro_ca).mul(dro_ca).mul(dro_ca).mul(dro_ca).mul(mask_neighbor_intra_soft) ##

                neighbor_sim = sim_sat_sat[:, indices1]
                p = F.softmax(neighbor_sim, dim=1)
                q = torch.ones_like(p) / p.size(1)
                epsilon = 1e-8
                p = (p + epsilon) / (1 + epsilon * p.size(1))
                consistency_loss = F.kl_div(p.log(), q, reduction='batchmean').clamp_min(0) 
                dro_dro_loss += 0.01 * consistency_loss
                dro_dro_loss = dro_dro_loss*0
                sim_prob_dro_dro_exp = sim_prob_dro_dro / 0.05  # 64*13638
                score_intra_dro_dro_s = F.softmax(sim_prob_dro_dro_exp, dim=1)
                score_intra_dro_dro_s = score_intra_dro_dro_s.clamp_min(1e-8)
                dro_dro_loss_s = -score_intra_dro_dro_s.log().mul(mask_neighbor_dro_dro).mul(mask_neighbor_prob_dro_dro).sum(
                    dim=1)  # .mul(sim_wise) mul(mask_neighbor_intra) .mul(mask_neighbor_intra)
                dro_dro_loss_s = lamda_s_neibor * dro_dro_loss_s.div(
                    num_neighbor_dro_dro).mean()  # .mul(dro_ca).mul(dro_ca).mul(mask_neighbor_intra_soft) ##
                neighbor_sim = sim_sat_sat[:, indices]
                p = F.softmax(neighbor_sim, dim=1)
                p = (p + epsilon) / (1 + epsilon * p.size(1))
                mutual_info = (p * p.log()).sum(dim=1).mean().clamp_min(0)
                dro_dro_loss_s += 0.1 * mutual_info
            if epoch >= self.cmlabel:
                loss = (loss_dro + loss_sat + loss_sat_s + loss_dro_s) + 0.1 * (
                            dro_dro_loss + sat_sat_loss + dro_dro_loss_s + sat_sat_loss_s) + 0.1 * (
                                   sat_dro_loss + dro_sat_loss + sat_dro_loss_s + dro_sat_loss_s)
            else:

                loss = (loss_dro + loss_sat + loss_sat_s + loss_dro_s) + 1 * (
                            dro_dro_loss + sat_sat_loss + dro_dro_loss_s + sat_sat_loss_s) + 0.5 * (
                                   sat_dro_loss + dro_sat_loss + sat_dro_loss_s + dro_sat_loss_s)
            with torch.no_grad():
                self.wise_memory_dro.updateEM(f_out_dro, index_dro)
                self.wise_memory_sat.updateEM(f_out_sat, index_sat)
                self.wise_memory_dro_s.updateEM(f_out_dro_s, index_dro)
                self.wise_memory_sat_s.updateEM(f_out_sat_s, index_sat)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.update(loss.item())

            loss_dro_log.update(loss_dro.item())
            loss_sat_log.update(loss_sat.item())
            dro_sat_loss_log.update(dro_sat_loss.item())
            dro_dro_loss_log.update(dro_dro_loss.item())
            sat_dro_loss_log.update(sat_dro_loss.item())
            # print log
            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:
                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      'Loss {:.3f} ({:.3f})\t'
                      'Loss dro {:.3f} ({:.3f})\t'
                      'Loss sat {:.3f} ({:.3f})\t'
                      'dro_sat_loss_log {:.3f} ({:.3f})\t'
                      'sat_dro_loss_log {:.3f} ({:.3f})\t'
                      'dro_dro_loss_log {:.3f} ({:.3f})\t'
                      'sat_sat_loss_log {:.3f} ({:.3f})\t'
                      .format(epoch, i + 1, len(data_loader_sat),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses.val, losses.avg, loss_dro_log.val, loss_dro_log.avg, loss_sat_log.val,
                              loss_sat_log.avg, \
                              dro_sat_loss_log.val, dro_sat_loss_log.avg, sat_dro_loss_log.val, sat_dro_loss_log.avg, \
                              dro_dro_loss_log.val, dro_dro_loss_log.avg, sat_sat_loss_log.val, sat_sat_loss_log.avg))
                print('loss_dro_s,loss_sat_s', loss_dro_s.item(), loss_sat_s.item())
                print('dro_sat_loss_s,sat_dro_loss_s', dro_sat_loss_s.item(), sat_dro_loss_s.item())
                print('dro_dro_loss_s,sat_sat_loss_s', dro_dro_loss_s.item(), sat_sat_loss_s.item())
                
    def _parse_data_sat(self, inputs):
        imgs, imgs1, name, pids, cids, indexes = inputs
        return imgs.cuda(), imgs1.cuda(), pids.cuda(), indexes.cuda(), cids.cuda(), name

    def _parse_data_dro(self, inputs):
        imgs, name, pids, cids, indexes = inputs
        return imgs.cuda(), pids.cuda(), indexes.cuda(), cids.cuda(), name

    def _forward(self, x1, x2):
        return self.encoder(x1, x2)
