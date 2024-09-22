# -*- coding: utf-8 -*-
import os
import torch
import tifffile
import numpy as np
import cv2
import JackFramework as jf


class DataReader(object):
    _DEPTH_DIVIDING = 256.0

    def __init__(self, args: object) -> object:
        self.__args = args
        self.__img_read_func, self.__label_read_func = \
            self.__read_func(args.dataset)

    def _get_img_read_func(self):
        return self.__img_read_func, self.__label_read_func

    def _read_data(self, left_img_path: str, right_img_path: str, gt_dsp_path: str) -> tuple:
        left_img = np.array(self.__img_read_func(left_img_path))
        right_img = np.array(self.__img_read_func(right_img_path))
        gt_dsp = np.array(self.__label_read_func(gt_dsp_path))
        return left_img, right_img, gt_dsp

    def _read_training_data(self, left_img_path: str, right_img_path: str,
                            gt_dsp_path: str) -> tuple:
        args = self.__args

        left_img, right_img, gt_dsp = self._read_data(left_img_path, right_img_path, gt_dsp_path)
        left_img, right_img = left_img[:, :, :3], right_img[:, :, :3]

        gt_dsp = gt_dsp if gt_dsp.ndim == 3 else np.expand_dims(gt_dsp, axis=2)
        height, width, _ = left_img.shape
        left_img, right_img, gt_dsp = jf.DataAugmentation.random_crop(
            [left_img, right_img, gt_dsp], width, height, args.imgWidth, args.imgHeight)

        gt_dsp = np.squeeze(gt_dsp, axis=2)

        left_img = jf.DataAugmentation.standardize(left_img)
        right_img = jf.DataAugmentation.standardize(right_img)

        left_img, right_img = left_img.transpose(2, 0, 1), right_img.transpose(2, 0, 1)
        gt_dsp[np.isinf(gt_dsp)] = 0

        return left_img, right_img, gt_dsp

    def _img_padding(self, left_img: np.array, right_img: np.array) -> tuple:
        # pading size
        args = self.__args

        if left_img.shape[0] < args.imgHeight:
            padding_height, padding_width = args.imgHeight, args.imgWidth
        else:
            padding_height, padding_width = \
                self._padding_size(left_img.shape[0]), self._padding_size(left_img.shape[1])

        top_pad, left_pad = padding_height - left_img.shape[0], padding_width - right_img.shape[1]

        if top_pad > 0 or left_pad > 0:
            # pading
            left_img = np.lib.pad(left_img, ((top_pad, 0), (0, left_pad), (0, 0)),
                                  mode='constant', constant_values=0)
            right_img = np.lib.pad(right_img, ((top_pad, 0), (0, left_pad), (0, 0)),
                                   mode='constant', constant_values=0)

        left_img, right_img = left_img.transpose(2, 0, 1), right_img.transpose(2, 0, 1)
        return left_img, right_img, top_pad, left_pad

    def _read_testing_data(self, left_img_path: str,
                           right_img_path: str,
                           gt_dsp_path: str) -> object:
        args = self.__args

        left_img = np.array(self.__img_read_func(left_img_path))
        right_img = np.array(self.__img_read_func(right_img_path))
        left_img, right_img = left_img[:, :, :3], right_img[:, :, :3]

        left_img = jf.DataAugmentation.standardize(left_img)
        right_img = jf.DataAugmentation.standardize(right_img)

        left_img, right_img, top_pad, left_pad = self._img_padding(left_img, right_img)

        if gt_dsp_path != 'None':
            gt_dsp = np.array(self.__label_read_func(gt_dsp_path))
            gt_dsp = gt_dsp if gt_dsp.ndim == 2 else np.squeeze(gt_dsp, axis=2)
            if top_pad > 0 or left_pad > 0:
                gt_dsp = np.lib.pad(gt_dsp, ((top_pad, 0), (0, left_pad)),
                                    mode='constant', constant_values=0)
        else:
            gt_dsp = np.zeros([left_img.shape[0], left_img.shape[1]])

        name = self._get_name(args.dataset, left_img_path)

        return left_img, right_img, gt_dsp, top_pad, left_pad, name

    def _get_img_read_func_dict(self) -> dict:
        return {'US3D': (tifffile.imread, tifffile.imread),
                'kitti2012': (jf.ImgIO.read_img, self._read_png_disp),
                'kitti2015': (jf.ImgIO.read_img, self._read_png_disp),
                'eth3d': (jf.ImgIO.read_img, self._read_pfm_disp),
                'middlebury': (jf.ImgIO.read_img, self._read_pfm_disp),
                'sceneflow': (jf.ImgIO.read_img, self._read_pfm_disp),
                'crestereo': (jf.ImgIO.read_img, self._read_cre_disp),
                'rob': (jf.ImgIO.read_img, self._read_rob_disp),
                'whu': (self._read_gray_tiff, self._read_gray_tiff),
                }

    def __read_func(self, dataset_name: str) -> object:
        func_dict = self._get_img_read_func_dict()
        assert dataset_name in func_dict
        return func_dict[dataset_name]

    def get_data(self, left_img_path: str, right_img_path: str,
                 gt_dsp_path: str, is_training: bool) -> tuple:
        if is_training:
            return self._read_training_data(left_img_path, right_img_path, gt_dsp_path)
        return self._read_testing_data(left_img_path, right_img_path, gt_dsp_path)

    @staticmethod
    def expand_batch_size_dims(left_img: np.array, right_img: np.array, gt_dsp: np.array,
                               top_pad: int, left_pad: int, name: str) -> tuple:
        left_img, right_img = np.expand_dims(left_img, axis=0), np.expand_dims(right_img, axis=0)
        gt_dsp, top_pad, left_pad, name = [gt_dsp], [top_pad], [left_pad], [name]
        left_img, right_img = torch.from_numpy(left_img), torch.from_numpy(right_img)
        return left_img, right_img, gt_dsp, top_pad, left_pad, name

    @staticmethod
    def _read_png_disp(path: str) -> np.array:
        gt_dsp = jf.ImgIO.read_img(path)
        gt_dsp = np.ascontiguousarray(
            gt_dsp, dtype=np.float32) / float(DataReader._DEPTH_DIVIDING)
        return gt_dsp

    @staticmethod
    def _read_cre_disp(path: str) -> np.array:
        gt_dsp = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        return gt_dsp.astype(np.float32) / 32

    @staticmethod
    def _read_pfm_disp(path: str) -> np.array:
        gt_dsp, _ = jf.ImgIO.read_pfm(path)
        return gt_dsp

    @staticmethod
    def _read_gray_tiff(path: str) -> np.array:
        return np.array(tifffile.imread(path)).expand_dims(img, axis=2)

    @staticmethod
    def _read_rob_disp(path: str) -> np.array:
        file_type = os.path.splitext(path)[-1]
        if file_type == ".png":
            gt_dsp = DataReader._read_png_disp(path)
        else:
            gt_dsp = DataReader._read_pfm_disp(path)
        return gt_dsp

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
        elif dataset_name in {"US3D"}:
            name = os.path.basename(path)
            pos = name.find('LEFT_RGB')
            name = name[:pos]
        return name
