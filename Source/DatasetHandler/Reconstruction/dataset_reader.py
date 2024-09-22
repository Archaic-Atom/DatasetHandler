# -*- coding: utf-8 -*-
import pandas as pd
from torch.utils.data.dataset import Dataset

try:
    from ._data_reader import DataReader
except ImportError:
    from _data_reader import DataReader


class ReconstructionDataset(Dataset):
    _DEPTH_DIVIDING = 256.0

    def __init__(self, args: object, list_path: str, is_training: bool = False) -> None:
        self.__args = args
        self.__is_training = is_training
        self.__data_handler = DataReader(self.__args)

        path_data = pd.read_csv(list_path)
        self.__img_path = path_data["img"].values

        self.__get_path = self._get_training_path if is_training else self._get_testing_path
        self.__data_steam = list(zip(self.__img_path))

    def _get_training_path(self, idx: int) -> list:
        return self.__img_path[idx]

    def _get_testing_path(self, idx: int) -> list:
        return self.__img_path[idx]

    def __getitem__(self, idx: int) -> tuple:
        return self.__data_handler.get_data(self.__get_path(idx), self.__is_training)

    def __len__(self) -> int:
        return len(self.__data_steam)
