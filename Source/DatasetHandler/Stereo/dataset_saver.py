# -*- coding: utf-8 -*-
import numpy as np

try:
    from ._data_saver import DataSaver
except ImportError:
    from _data_saver import DataSaver


class StereoSaver(DataSaver):
    """docstring for StereoSaver"""
    _DEPTH_DIVIDING = 256.0

    def __init__(self, args: object) -> object:
        super().__init__(args)
        self.__args = args

    def save_output(self, disp: np.array, img_id: int, dataset_name: str,
                    supplement: list, ttimes: float) -> None:
        bs, _, _ = disp.shape
        top_pads, left_pads, names = supplement[0], supplement[1], supplement[2]

        for i in range(bs):
            per_disp, top_pad, left_pad = disp[i, :, :], top_pads[i], left_pads[i]
            per_disp = self._crop_test_img(per_disp, top_pad, left_pad)
            self._save_per_output(i, bs, per_disp, img_id, dataset_name, names, ttimes)

    def save_output_by_path(self, disp: np.array, supplement: list, path: str) -> None:
        bs, _, _ = disp.shape
        assert bs == 1
        top_pads, left_pads = supplement[0], supplement[1]
        for i in range(bs):
            per_disp, top_pad, left_pad = disp[i, :, :], top_pads[i], left_pads[i]
            per_disp = self._crop_test_img(per_disp, top_pad, left_pad)
            per_disp = self._depth2img(per_disp)
            self._save_png_img(path, per_disp)
