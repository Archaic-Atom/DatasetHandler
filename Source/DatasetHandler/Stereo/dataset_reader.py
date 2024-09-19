# -*- coding: utf-8 -*-
import pandas as pd
from torch.utils.data.dataset import Dataset

try:
    from ._data_reader import DataReader
except ImportError:
    from _data_reader import DataReader


class StereoDataset(Dataset):
    _DEPTH_DIVIDING = 256.0

    def __init__(self, args: object, list_path: str, is_training: bool = False) -> None:
        self.__args = args
        self.__is_training = is_training
        self.__data_handler = DataReader(self.__args)

        path_data = pd.read_csv(list_path)
        self.__left_img_path, self.__right_img_path, self.__gt_disp_path =\
            path_data["left_img"].values, path_data["right_img"].values, path_data["gt_disp"].values

        self.__get_path = self._get_training_path if is_training else self._get_testing_path
        self.__data_steam = list(zip(
            self.__left_img_path, self.__right_img_path, self.__gt_disp_path)) if is_training \
            else list(zip(self.__left_img_path, self.__right_img_path))

    def _get_training_path(self, idx: int) -> list:
        return self.__left_img_path[idx], self.__right_img_path[idx], self.__gt_disp_path[idx]

    def _get_testing_path(self, idx: int) -> list:
        return self.__left_img_path[idx], self.__right_img_path[idx], self.__gt_disp_path[idx]

    def __getitem__(self, idx: int) -> tuple:
        left_img_path, right_img_path, gt_dsp_path = self.__get_path(idx)
        return self.__data_handler.get_data(
            left_img_path, right_img_path, gt_dsp_path, self.__is_training)

    def get_data(self, left_img_path, right_img_path, gt_dsp_path, is_training) -> tuple:
        return self.__data_handler.get_data(
            left_img_path, right_img_path, gt_dsp_path, is_training)

    def expand_batch_size_dims(self, left_img, right_img, gt_dsp, top_pad, left_pad, name) -> tuple:
        return self.__data_handler.expand_batch_size_dims(
            left_img, right_img, gt_dsp, top_pad, left_pad, name)

    def __len__(self) -> int:
        return len(self.__data_steam)
