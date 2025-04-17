def test(epoch):
    net.eval()
    print('Extracting Gallery Feature...')
    start = time.time()

    # 先用一批数据动态获取 feature dim
    with torch.no_grad():
        sample_input = next(iter(gall_loader))[0].cuda()
        sample_feat = net(sample_input, sample_input)
        chunk_dim = sample_feat.shape[1] // 5  # 每个分块的维度
        pool_dim = chunk_dim

    gall_feat1 = np.zeros((ngall, pool_dim))
    gall_feat2 = np.zeros((ngall, pool_dim))
    gall_feat3 = np.zeros((ngall, pool_dim))
    gall_feat4 = np.zeros((ngall, pool_dim))
    gall_feat5 = np.zeros((ngall, pool_dim))

    ptr = 0
    with torch.no_grad():
        for batch_idx, (input, label) in enumerate(gall_loader):
            batch_num = input.size(0)
            input = input.cuda()
            feat = net(input, input)  # 输出 shape: (B, 5120)
            chunked = torch.chunk(feat, 5, dim=1)
            for i, chunk in enumerate(chunked):
                chunk_np = chunk.detach().cpu().numpy()
                if i == 0:
                    gall_feat1[ptr:ptr+batch_num] = chunk_np
                elif i == 1:
                    gall_feat2[ptr:ptr+batch_num] = chunk_np
                elif i == 2:
                    gall_feat3[ptr:ptr+batch_num] = chunk_np
                elif i == 3:
                    gall_feat4[ptr:ptr+batch_num] = chunk_np
                elif i == 4:
                    gall_feat5[ptr:ptr+batch_num] = chunk_np
            ptr += batch_num
    print('Gallery Extracting Time:\t {:.3f}'.format(time.time() - start))

    print('Extracting Query Feature...')
    start = time.time()
    query_feat1 = np.zeros((nquery, pool_dim))
    query_feat2 = np.zeros((nquery, pool_dim))
    query_feat3 = np.zeros((nquery, pool_dim))
    query_feat4 = np.zeros((nquery, pool_dim))
    query_feat5 = np.zeros((nquery, pool_dim))

    ptr = 0
    with torch.no_grad():
        for batch_idx, (input, label) in enumerate(query_loader):
            batch_num = input.size(0)
            input = input.cuda()
            feat = net(input, input)
            chunked = torch.chunk(feat, 5, dim=1)
            for i, chunk in enumerate(chunked):
                chunk_np = chunk.detach().cpu().numpy()
                if i == 0:
                    query_feat1[ptr:ptr+batch_num] = chunk_np
                elif i == 1:
                    query_feat2[ptr:ptr+batch_num] = chunk_np
                elif i == 2:
                    query_feat3[ptr:ptr+batch_num] = chunk_np
                elif i == 3:
                    query_feat4[ptr:ptr+batch_num] = chunk_np
                elif i == 4:
                    query_feat5[ptr:ptr+batch_num] = chunk_np
            ptr += batch_num
    print('Query Extracting Time:\t {:.3f}'.format(time.time() - start))

    # compute similarity
    print("Computing Distance...")
    start = time.time()
    distmat1 = np.matmul(query_feat1, gall_feat1.T)
    distmat2 = np.matmul(query_feat2, gall_feat2.T)
    distmat3 = np.matmul(query_feat3, gall_feat3.T)
    distmat4 = np.matmul(query_feat4, gall_feat4.T)
    distmat5 = np.matmul(query_feat5, gall_feat5.T)
    distmat7 = distmat1 + distmat2 + distmat3 + distmat4 + distmat5
    print('Distance Computation Time:\t {:.3f}'.format(time.time() - start))

    # 这里继续执行评估（例子如下）
    if dataset == 'regdb':
        cmc1, mAP1, mINP1 = eval_regdb(-distmat1, query_label, gall_label)
        cmc7, mAP7, mINP7 = eval_regdb(-distmat7, query_label, gall_label)
    elif dataset == 'sysu':
        cmc1, mAP1, mINP1 = eval_sysu(-distmat1, query_label, gall_label, query_cam, gall_cam)
        cmc7, mAP7, mINP7 = eval_sysu(-distmat7, query_label, gall_label, query_cam, gall_cam)
    elif dataset == 'llcm':
        cmc1, mAP1, mINP1 = eval_llcm(-distmat1, query_label, gall_label, query_cam, gall_cam)
        cmc7, mAP7, mINP7 = eval_llcm(-distmat7, query_label, gall_label, query_cam, gall_cam)

    return cmc1, mAP1, mINP1, None, None, None, cmc7, mAP7, mINP7
