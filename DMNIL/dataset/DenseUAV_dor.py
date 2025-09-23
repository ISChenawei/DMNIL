from __future__ import print_function, absolute_import
import os.path as osp
import glob
import re
from ..Utils.base_dataset import BaseImageDataset


class DenseUAV_dor(BaseImageDataset):
    """
    Market1501
    Reference:
    Zheng et al. Scalable Person Re-identification: A Benchmark. ICCV 2015.
    URL: http://www.liangzheng.org/Project/project_reid.html

    Dataset statistics:
    # identities: 1501 (+1 for background)
    # images: 12936 (train) + 3368 (query) + 15913 (gallery)
    """
    dataset_dir = 'DenseUAV'

    def __init__(self, root, verbose=True, **kwargs):
        super(DenseUAV_dor, self).__init__()
        root='sample4geo/dataset'
        # print('regdb_rgb',trial)
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.train_dir = osp.join(self.dataset_dir, 'train/drone')
        self.query_dir = osp.join(self.dataset_dir, 'test/query_drone')
        self.gallery_dir = osp.join(self.dataset_dir, 'test/gallery_drone')
        self._check_before_run()

        train = self._process_dir(self.train_dir, relabel=True)
        query = self._process_dir(self.query_dir, relabel=False)
        gallery = self._process_dir(self.gallery_dir, relabel=False)

        if verbose:
            print("=> DenseUAV_dor loaded")
            self.print_dataset_statistics(train, query, gallery)

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids, self.num_train_imgs, self.num_train_cams = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams = self.get_imagedata_info(self.gallery)

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not osp.exists(self.query_dir):
            raise RuntimeError("'{}' is not available".format(self.query_dir))
        if not osp.exists(self.gallery_dir):
            raise RuntimeError("'{}' is not available".format(self.gallery_dir))

    def _process_dir(self, dir_path, relabel=False):
        img_paths = glob.glob(osp.join(dir_path, '**', '*.*'), recursive=True)  # 匹配所有扩展名
        img_paths = [p for p in img_paths if osp.splitext(p)[-1].lower() in ['.jpg', '.jpeg', '.png','tif']]  # 过滤支持的扩展名
        # print(f"Filtered images: {img_paths}")
        # print(f"Found {len(img_paths)} images in {dir_path}")
        dataset = []
        pid_container = set()

        for img_path in img_paths:
            # 提取 person ID（子文件夹名）
            pid = osp.basename(osp.dirname(img_path))  # 获取上一级文件夹名作为 person ID
            if not pid.isdigit():
                print(f"Warning: Skipping file with invalid PID: {img_path}")
                continue  # 跳过无效 person ID 的文件

            pid = int(pid)
            if pid == -1:
                continue  # 忽略背景图片
            pid_container.add(pid)

        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        for img_path in img_paths:
            pid = osp.basename(osp.dirname(img_path))
            if not pid.isdigit():
                continue
            pid = int(pid)
            if pid == -1:
                continue
            camid = 1  # 默认摄像头 ID 为 0（可根据实际需要修改）
            if relabel:
                pid = pid2label[pid]
            dataset.append((img_path, pid, camid))

        return dataset