# -*- coding: utf-8 -*-
import cv2
import torch
import tifffile
import numpy as np
import JackFramework as jf

try:
    from .mask_aug import MaskAug
except ImportError:
    from mask_aug import MaskAug


class DataReader(object):
    _DEPTH_DIVIDING = 256.0

    def __init__(self, args: object) -> object:
        self.__args = args
        self.__img_read_func, self.__label_read_func = self.__read_func(args.dataset)
        self.mask_aug = MaskAug(args.imgHeight, args.imgWidth,
                                args.block_size, args.mask_ratio)

    def _get_img_read_func(self):
        return self.__img_read_func

    def _read_data(self, img_path: str) -> tuple:
        return np.array(self.__img_read_func(img_path))

    def _read_label(self, img_path: str) -> tuple:
        return np.array(self.__label_read_func(img_path))

    def _read_training_data(self, img_path: str) -> tuple:
        args = self.__args

        img = self._read_data(img_path)
        gt_img = self._read_label(img_path)

        # img, gt_img = jf.DataAugmentation.random_scale([img, gt_img])
        if len(img.shape) == 2:
            img, gt_img = np.expand_dims(img, axis=2), np.expand_dims(gt_img, axis=2)
        height, width, _ = img.shape
        img, gt_img = jf.DataAugmentation.random_crop(
            [img, gt_img], width, height, args.imgWidth, args.imgHeight)
        img, gt_img = jf.DataAugmentation.random_horizontal_flip([img, gt_img])
        org_img = img.copy()
        img, mask, mask_img_patch, random_sample_list = self.mask_aug(img)
        # new_img = self.mask_aug.reconstruct_img(img, mask_img_patch, random_sample_list)
        # mask_img_patch = jf.DataAugmentation.standardize(mask_img_patch)
        for i in range(img.shape[2]):
            gt_img[:, :, i] = gt_img[:, :, i] * (1 - mask)

        img, gt_img = img.transpose(2, 0, 1), gt_img.transpose(2, 0, 1)
        mask_img_patch = mask_img_patch.transpose(3, 2, 0, 1)
        random_sample_list = np.array(random_sample_list)
        img, mask_img_patch, gt_img = img.copy(), mask_img_patch.copy(), gt_img.copy()
        org_img = org_img.transpose(2, 0, 1)
        return org_img.astype(np.float32), mask_img_patch.astype(np.float32),\
            random_sample_list, gt_img

    def _img_padding(self, img: np.array) -> tuple:
        # pading size
        args = self.__args

        if img.shape[0] < args.imgHeight and img.shape[1] < args.imgWidth:
            padding_height, padding_width = args.imgHeight, args.imgWidth
        else:
            padding_height, padding_width = \
                self._padding_size(img.shape[0]), self._padding_size(img.shape[1])

        top_pad, left_pad = padding_height - img.shape[0], padding_width - img.shape[1]

        assert(top_pad >= 0 and left_pad >= 0)
        # pading
        if top_pad > 0 or left_pad > 0:
            img = np.lib.pad(img, ((top_pad, 0), (0, left_pad), (0, 0)),
                             mode='constant', constant_values=0)

        return img, top_pad, left_pad

    def _read_testing_data(self, img_path: str) -> tuple:
        args = self.__args
        img = np.array(self.__img_read_func(img_path))
        if len(img.shape) == 2:
            img = np.expand_dims(img, axis=2)

        img, top_pad, left_pad = self._img_padding(img)
        img, mask, mask_img_patch, random_sample_list = self.mask_aug(img)
        # new_img = self.mask_aug.reconstruct_img(img, mask_img_patch, random_sample_list)
        # mask_img_patch = jf.DataAugmentation.standardize(mask_img_patch)
        name = self._get_name(args.dataset, img_path)

        img, mask_img_patch = img.transpose(2, 0, 1), mask_img_patch.transpose(3, 2, 0, 1)
        random_sample_list = np.array(random_sample_list)
        return img.astype(np.float32), mask_img_patch.astype(np.float32),\
            random_sample_list, top_pad, left_pad, name, mask, img

    def __read_func(self, dataset_name: str) -> object:
        img_read_func, label_read_func = None, None
        for case in jf.Switch(dataset_name):
            if case('US3D'):
                img_read_func, label_read_func = self._read_normal_tiff, self._read_normal_tiff
                break
            if case('whu'):
                img_read_func, label_read_func = self._read_gray_tiff, self._read_gray_tiff
                break
            if case():
                jf.log.error("The dataset's name is error!!!")

        return img_read_func, label_read_func

    def get_data(self, img_path: str, is_training: bool) -> tuple:
        if is_training:
            return self._read_training_data(img_path)
        return self._read_testing_data(img_path)

    @staticmethod
    def expand_batch_size_dims(img: np.array, mask_img_patch: np.array, random_sample_list: list,
                               top_pad: int, left_pad: int, name: str, mask: np.array) -> tuple:
        img = torch.from_numpy(np.expand_dims(img, axis=0))
        mask_img_patch = torch.from_numpy(np.expand_dims(mask_img_patch, axis=0))
        top_pad, left_pad, name = [top_pad], [left_pad], [name]
        mask = torch.from_numpy(np.expand_dims(img, axis=0))
        random_sample_list = torch.from_numpy(np.expand_dims(random_sample_list, axis=0))
        return img, mask_img_patch, random_sample_list, top_pad, left_pad, name, mask

    @staticmethod
    def _read_gray_tiff(path: str) -> np.array:
        img = np.array(tifffile.imread(path))
        return (img - img.min()) / (img.max() - img.min())

    @staticmethod
    def _read_normal_tiff(path: str) -> np.array:
        img = np.array(tifffile.imread(path))
        return img / 255.0

    @staticmethod
    def _padding_size(value: int, base: int = 64) -> int:
        off_set = 1
        return value // base + off_set

    @staticmethod
    def _get_name(dataset_name: str, path: str) -> str:
        name = ""
        if dataset_name in {"eth3d", "middlebury"}:
            off_set = 1
            pos = path.rfind('/')
            name = path[:pos]
            pos = name.rfind('/')
            name = name[pos + off_set:]
        return name
