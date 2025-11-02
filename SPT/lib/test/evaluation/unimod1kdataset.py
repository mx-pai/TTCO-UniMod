import os
import numpy as np
from lib.test.evaluation.data import Sequence, BaseDataset, SequenceList
from lib.test.utils.load_text import load_text
from glob import glob
import cv2

class UniMod1KDataset(BaseDataset):
    def __init__(self):
        super().__init__()
        self.base_path = self.env_settings.unimod1k_path
        self.sequence_list = self._get_sequence_list()

    def get_sequence_list(self):
        return SequenceList([self._construct_sequence(s) for s in self.sequence_list])

    def _construct_sequence(self, sequence_name):
        sequence_path = sequence_name
        nz = 8

        # 1️⃣ 读取 groundtruth.txt（只第一帧）
        anno_path = os.path.join(self.base_path, sequence_name, 'groundtruth.txt')
        try:
            gt = np.loadtxt(anno_path, dtype=np.float64)
        except:
            gt = np.loadtxt(anno_path, delimiter=',', dtype=np.float64)

        gt = np.array(gt, dtype=np.float64).reshape(1, 4)  # 只保留第一帧矩形框

        # 2️⃣ NLP 文本
        nlp_path = os.path.join(self.base_path, sequence_name, 'nlp.txt')
        nlp_label = ""
        if os.path.exists(nlp_path):
            nlp_label = load_text(str(nlp_path), delimiter=',', dtype=str)
            nlp_label = str(nlp_label)

        # 3️⃣ 根据 color 文件夹中的帧数决定视频长度
        color_dir = os.path.join(self.base_path, sequence_name, 'color')
        depth_dir = os.path.join(self.base_path, sequence_name, 'depth')
        color_imgs = sorted(glob(os.path.join(color_dir, '*.png')))
        depth_imgs = sorted(glob(os.path.join(depth_dir, '*.png')))

        num_frames = min(len(color_imgs), len(depth_imgs))
        if num_frames == 0:
            raise RuntimeError(f"No frames found in {sequence_name}")

        # 4️⃣ 构造帧列表（从第 1 帧开始）
        frames = []
        for i in range(num_frames):
            c_path = os.path.join(color_dir, f"{i+1:0{nz}}.png")
            d_path = os.path.join(depth_dir, f"{i+1:0{nz}}.png")
            frames.append({'color': c_path, 'depth': d_path})

        # 5️⃣ 直接返回 Sequence（只用第一帧 GT 初始化）
        return Sequence(sequence_name, frames, 'unimod1k', gt, language_query=nlp_label)

    def __len__(self):
        return len(self.sequence_list)

    def _get_sequence_list(self):
        list_file = os.path.join(self.base_path, 'list.txt')
        with open(list_file, 'r') as f:
            sequence_list = f.read().splitlines()
        return sequence_list
