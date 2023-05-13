# -*- coding: utf-8 -*-
import random
import numpy as np


class MaskAug(object):
    """docstring for ClassName"""
    COLOR_GRAY = 0

    def __init__(self, img_height: int, img_width: int,
                 block_size: int = 2, ratio: float = 0.3, ) -> None:
        super().__init__()
        assert((img_height % block_size == 0) and (img_width % block_size == 0))
        self._img_height, self._img_width, self._block_size = img_height, img_width, block_size
        self._block_width_num = int(img_width / block_size)
        self._block_height_num = int(img_height / block_size)
        self._block_num = int(img_height * img_width / block_size / block_size)
        self._block_num_list = list(range(0, self._block_num))
        self._sample_num = int(self._block_num * (1 - ratio))

    def _generate_mask(self, img: np.array = None) -> np.array:
        random_sample_list = random.sample(self._block_num_list, self._sample_num)
        mask = np.zeros([self._img_height, self._img_width], dtype = float)
        mask_img_patch = []
        for sample_id in random_sample_list:
            height_id = sample_id // self._block_height_num
            width_id = sample_id % self._block_width_num
            cy = height_id * self._block_size
            cx = width_id * self._block_size
            mask_img_patch.append(img[cy:cy + self._block_size, cx:cx + self._block_size, :].copy())
            mask[cy:cy + self._block_size, cx:cx + self._block_size] = 1
        return mask, np.array(mask_img_patch).transpose(1, 2, 3, 0), random_sample_list

    def __call__(self, img: np.array) -> np.array:
        mask, mask_img_patch, random_sample_list = self._generate_mask(img)
        for i in range(img.shape[2]):
            img[:, :, i] = img[:, :, i] * mask + (1 - mask) * MaskAug.COLOR_GRAY
        return img, mask, mask_img_patch, random_sample_list

    @staticmethod
    def reconstruct_img(img: np.array, mask_img_patch: np.array,
                        random_sample_list: list) -> np.array:
        h, w, c = img.shape
        block_h, block_w, _, _ = mask_img_patch.shape
        block_width_num, block_height_num = int(h / block_w), int(w / block_h)
        new_img = np.zeros([h, w, c], dtype = float)
        for idx, sample_id in enumerate(random_sample_list):
            height_id = sample_id // block_height_num
            width_id = sample_id % block_width_num
            new_img[height_id * block_h:height_id * block_h + block_h,
                    width_id * block_w: width_id * block_w + block_w, :] =\
                mask_img_patch[:, :, :, idx]
        return new_img
