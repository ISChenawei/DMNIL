import os.path
import scipy.io
import torch
import numpy as np
from tqdm import tqdm
import gc
from ..trainer import predict
def evaluate(config,
             model,
             query_loader,
             gallery_loader,
             ranks=[1, 5, 10],
             step_size=1000,
             cleanup=True):
    print("Extract Features:")
    img_features_gallery, ids_gallery,paths_gallery = predict(config, model, gallery_loader)
    img_features_query, ids_query,paths_query = predict(config, model, query_loader)

    gl = ids_gallery.cpu().numpy()
    ql = ids_query.cpu().numpy()
    save_features_to_mat(img_features_query,img_features_gallery,ql,gl,paths_query, paths_gallery)
    print("Compute Scores:")
    CMC = torch.IntTensor(len(ids_gallery)).zero_()
    ap = 0.0
    for i in tqdm(range(len(ids_query))):
        ap_tmp, CMC_tmp = eval_query(img_features_query[i], ql[i],img_features_gallery,gl ,paths_query[i],paths_gallery )
        if CMC_tmp[0] == -1:
            continue
        CMC = CMC + CMC_tmp
        ap += ap_tmp

    AP = ap / len(ids_query) * 100

    CMC = CMC.float()
    CMC = CMC / len(ids_query)  # average CMC

    # top 1%
    top1 = round(len(ids_gallery) * 0.01)

    string = []

    for i in ranks:
        string.append('Recall@{}: {:.4f}'.format(i, CMC[i - 1] * 100))

    string.append('Recall@top1: {:.4f}'.format(CMC[top1] * 100))
    string.append('AP: {:.4f}'.format(AP))

    print(' - '.join(string))

    # cleanup and free memory on GPU
    if cleanup:
        del img_features_query, ids_query, img_features_gallery, ids_gallery
        gc.collect()
        # torch.cuda.empty_cache()

    return CMC[0]


def eval_query(qf, ql, gf, gl,query_path,paths_gallery):
    score = gf @ qf.unsqueeze(-1)
    top_k = 10
    score = score.squeeze().cpu().numpy()

    # predict index
    index = np.argsort(score)  # from small to large
    index = index[::-1]

    # good index
    query_index = np.argwhere(gl == ql)
    good_index = query_index

    # junk index
    junk_index = np.argwhere(gl == -1)

    ## T-tsne
    CMC_tmp = compute_mAP(index, good_index, junk_index)
    return CMC_tmp

def save_features_to_mat(query_features, gallery_features, query_labels, gallery_labels, paths_query, paths_gallery, save_path='SUES-D-S-150-features.mat'):
    """
    保存提取的特征和标签到 .mat 文件
    """
    result = {
        'query_features': query_features.cpu().numpy(),
        'gallery_features': gallery_features.cpu().numpy(),
        'query_labels': query_labels,
        'gallery_labels': gallery_labels,
        'paths_query': paths_query,
        'paths_gallery': paths_gallery
    }
    scipy.io.savemat(save_path, result)
    print(f'Features and labels saved to {save_path}')

def compute_mAP(index, good_index, junk_index):
    ap = 0
    cmc = torch.IntTensor(len(index)).zero_()
    if good_index.size == 0:  # if empty
        cmc[0] = -1
        return ap, cmc

    # remove junk_index
    mask = np.in1d(index, junk_index, invert=True)
    index = index[mask]

    # find good_index index
    ngood = len(good_index)
    mask = np.in1d(index, good_index)
    rows_good = np.argwhere(mask == True)
    rows_good = rows_good.flatten()

    cmc[rows_good[0]:] = 1
    for i in range(ngood):
        d_recall = 1.0 / ngood
        precision = (i + 1) * 1.0 / (rows_good[i] + 1)
        if rows_good[i] != 0:
            old_precision = i * 1.0 / rows_good[i]
        else:
            old_precision = 1.0
        ap = ap + d_recall * (old_precision + precision) / 2

    return ap, cmc
