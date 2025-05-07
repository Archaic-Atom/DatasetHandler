# -*- coding: utf-8 -*-
import pandas as pd
from torch.utils.data.dataset import Dataset

try:
    from ._data_reader import DataReader
except ImportError:
    from _data_reader import DataReader


class ChangeDetectionDataset(Dataset):
    _LABLE_DIVIDING = 255.0
    _DATA_STEAM_LENGTH = 1

    def __init__(self, args: object, list_path: str, is_training: bool = False) -> None:
        self.__args = args
        self.__is_training = is_training
        self.__data_handler = DataReader(self.__args)

        path_data = pd.read_csv(list_path)
        self.__t1_img_path = path_data["T1"].values
        self.__t2_img_path = path_data["T2"].values
        self.__gt_path = path_data["gt"].values

        self.__get_path = self._get_training_path if is_training else self._get_testing_path
        self.__data_steam = list(zip(self.__t1_img_path, self.__t2_img_path, self.__gt_path))

    def _get_training_path(self, idx: int) -> list:
        return self.__t1_img_path[idx], self.__t2_img_path[idx], self.__gt_path[idx]

    def _get_testing_path(self, idx: int) -> list:
        return self.__t1_img_path[idx], self.__t2_img_path[idx], self.__gt_path[idx]

    def __getitem__(self, idx: int) -> tuple:
        idx = 0 if idx > len(self.__data_steam) else idx
        t1_img_path, t2_img_path, gt_path = self.__get_path(idx)
        return self.__data_handler.get_data(
            t1_img_path, t2_img_path, gt_path, self.__is_training)

    def __len__(self) -> int:
        # warning the code will need to change
        if len(self.__data_steam) > self._DATA_STEAM_LENGTH:
            return len(self.__data_steam)

        args = self.__args
        if self.__is_training is True:
            return args.imgNum
        return args.valImgNum
