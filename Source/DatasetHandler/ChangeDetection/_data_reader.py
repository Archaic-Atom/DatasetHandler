# -*- coding: utf-8 -*-
import cv2
import torch
import imageio
import tifffile
import numpy as np
import JackFramework as jf

try:
    from ._data_aug import random_crop_new, random_fliplr, random_flipud, random_rot, normalize_img
except ImportError:
    from _data_aug import random_crop_new, random_fliplr, random_flipud, random_rot, normalize_img


class DataReader(object):
    _DEPTH_DIVIDING = 256.0

    def __init__(self, args: object) -> object:
        self.__args = args
        self.__img_read_func, self.__label_read_func = self.__read_func(args.dataset)
        assert args.imgWidth == args.imgHeight
        self.crop_size = args.imgWidth

    def get_data(self, t1_img_path: str, t2_img_path: str, gt_path: str, is_training: bool) -> tuple:
        if is_training:
            return self._read_training_data(t1_img_path, t2_img_path, gt_path)
        return self._read_testing_data(t1_img_path, t2_img_path, gt_path)

    def _get_img_read_func(self):
        return self.__img_read_func

    def _read_data(self, img_path: str) -> tuple:
        return np.array(self.__img_read_func(img_path))

    def _read_label(self, img_path: str) -> tuple:
        return np.array(self.__label_read_func(img_path))

    def _read_training_data(self, t1_img_path: str, t2_img_path: str, gt_path: str) -> tuple:
        # args = self.__args
        t1_img = self._read_data(t1_img_path)
        t2_img = self._read_data(t2_img_path)
        gt_img = self._read_label(gt_path)
        return self.__transforms(t1_img, t2_img, gt_img, True)

    def _read_testing_data(self, t1_img_path: str, t2_img_path: str, gt_path: str) -> tuple:
        t1_img = self._read_data(t1_img_path)
        t2_img = self._read_data(t2_img_path)
        gt_img = self._read_label(gt_path)
        return self.__transforms(t1_img, t2_img, gt_img, False)

    def __read_func(self, dataset_name: str) -> object:
        img_read_func, label_read_func = None, None
        for case in jf.Switch(dataset_name):
            if case('WHU'):
                img_read_func, label_read_func = self._read_gray_tiff, self._read_normal_tiff
                break
            if case('LEVIR-CD'):
                img_read_func, label_read_func = self._read_png_img, self._read_png_img
                break
            if case():
                jf.log.error("The dataset's name is error!!!")
        return img_read_func, label_read_func

    def __transforms(self, t1_img: np.array,
                     t2_img: np.array,
                     gt_img: np.array, aug: bool = True) -> tuple:
        if aug:
            t1_img, t2_img, gt_img = random_crop_new(t1_img, t2_img, gt_img, self.crop_size)
            t1_img, t2_img, gt_img = random_fliplr(t1_img, t2_img, gt_img)
            t1_img, t2_img, gt_img = random_flipud(t1_img, t2_img, gt_img)
            t1_img, t2_img, gt_img = random_rot(t1_img, t2_img, gt_img)

        t1_img = self._normalize_img(t1_img)
        t2_img = self._normalize_img(t2_img)
        return t1_img, t2_img, gt_img

    @staticmethod
    def _read_gray_tiff(path: str) -> np.array:
        img = np.array(tifffile.imread(path))
        return (img - img.min()) / (img.max() - img.min())

    @staticmethod
    def _read_normal_tiff(path: str) -> np.array:
        img = np.array(tifffile.imread(path))
        return img / 255.0

    @staticmethod
    def _read_png_img(path: str) -> np.array:
        return np.array(imageio.imread(path), np.float32)

    @staticmethod
    def _normalize_img(img) -> np.array:
        img = normalize_img(img)
        return np.transpose(img, (2, 0, 1))
