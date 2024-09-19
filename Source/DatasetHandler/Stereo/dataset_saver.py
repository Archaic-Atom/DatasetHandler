# -*- coding: utf-8 -*-
import numpy as np

try:
    from ._data_saver import DataSaver
except ImportError:
    from _data_saver import DataSaver


class StereoSaver(DataSaver):
    """docstring for StereoSaver"""
    _DEPTH_DIVIDING = 256.0
    DIM_CHANNLES, DIM_NUM_BHW, DIM_NUM_BCHW = 1, 3, 4

    def __init__(self, args: object) -> object:
        super().__init__(args)
        self.__args = args

    def reduce_dim(self, disp: np.array) -> np.array:
        assert self.DIM_NUM_BHW == len(disp.shape) or \
            self.DIM_NUM_BCHW == len(disp.shape)
        if self.DIM_NUM_BCHW == len(disp.shape):
            disp = np.squeeze(disp, axis=self.DIM_CHANNLES)
        return disp

    def save_output(self, disp: np.array, img_id: int, dataset_name: str,
                    supplement: list, ttimes: float) -> None:
        disp = self.reduce_dim(disp)
        bs, _, _ = disp.shape
        _, top_pads, left_pads, names = supplement

        for i in range(bs):
            per_disp, top_pad, left_pad = disp[i, :, :], top_pads[i], left_pads[i]
            per_disp = self._crop_test_img(per_disp, top_pad, left_pad)
            self._save_per_output(i, bs, per_disp, img_id, dataset_name, names, ttimes)

    def save_output_by_path(self, disp: np.array, supplement: list, path: str) -> None:
        disp = self.reduce_dim(disp)
        bs, _, _ = disp.shape
        assert bs == 1
        _, top_pads, left_pads, _ = supplement

        for i in range(bs):
            per_disp, top_pad, left_pad = disp[i, :, :], top_pads[i], left_pads[i]
            per_disp = self._crop_test_img(per_disp, top_pad, left_pad)
            per_disp = self._depth2img(per_disp)
            self._save_png_img(path, per_disp)
