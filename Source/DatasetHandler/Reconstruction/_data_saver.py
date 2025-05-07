# -*- coding: utf-8 -*-
import numpy as np
import cv2
import JackFramework as jf
import tifffile


class DataSaver(object):
    """docstring for StereoSaver"""
    _DEPTH_DIVIDING = 256.0

    def __init__(self, args: object) -> object:
        super().__init__()
        self.__args = args

    def _save_per_output(self, idx: int, batch_size: int, tmp_res: np.array,
                         img_id: int, dataset_name: str, names: list, ttimes: float) -> None:
        for case in jf.Switch(dataset_name):
            if case('US3D'):
                name = batch_size * img_id + idx
                self.save_kitti_test_data(tmp_res, name)
                break
            if case('kitti2012') or case('kitti2015') or case('sceneflow'):
                name = batch_size * img_id + idx
                self.save_kitti_test_data(tmp_res, name)
                break
            if case('sceneflow'):
                name = batch_size * img_id + idx
                self.save_kitti_test_data(tmp_res, name)
                break
            if case('eth3d'):
                name = names[idx]
                self.save_eth3d_test_data(tmp_res, name, ttimes)
                break
            if case('middlebury'):
                name = names[idx]
                self.save_middlebury_test_data(tmp_res, name, ttimes)
                break
            if case('whu'):
                name = batch_size * img_id + idx
                self.save_whu_test_data(tmp_res, name)
                break
            if case('mask'):
                name = batch_size * img_id + idx
                self.save_mask_img(tmp_res, name)
                break
            if case():
                jf.log.error("The model's name is error!!!")

    def save_kitti_test_data(self, img: np.array, num: int) -> None:
        args = self.__args
        path = self._generate_output_img_path(args.resultImgDir, num)
        img = self._depth2img(img)
        self._save_png_img(path, img)

    def save_eth3d_test_data(self, img: np.array,
                             name: str, ttimes: str) -> None:
        args = self.__args
        path = args.resultImgDir + name + '.pfm'
        jf.ImgIO.write_pfm(path, img)
        path = args.resultImgDir + name + '.txt'
        with open(path, 'w') as f:
            f.write("runtime " + str(ttimes))
            f.close()

    def save_middlebury_test_data(self, img: np.array,
                                  name: str, ttimes: str) -> None:
        args = self.__args
        folder_name = args.resultImgDir + name + '/'
        jf.FileHandler.mkdir(folder_name)
        method_name = "disp0" + args.modelName + "_RVC.pfm"
        path = folder_name + method_name
        jf.ImgIO.write_pfm(path, img)

        time_name = "time" + args.modelName + "_RVC.txt"
        path = folder_name + time_name
        with open(path, 'w') as f:
            f.write(str(ttimes))
            f.close()

    def save_whu_test_data(self, img: np.array, num: int) -> None:
        args = self.__args
        path = self._generate_output_img_path(args.resultImgDir, num, img_type='.png')
        cv2.imwrite(path, img * 255)
        #tifffile.imsave(path, img * 255, compress=6)

    def save_mask_img(self, img: np.array, num: int) -> None:
        args = self.__args
        path = self._generate_output_img_path(args.resultImgDir, num)
        self._save_png_img(path, img.astype(np.uint8))

    @staticmethod
    def _crop_test_img(img: np.array, top_pad: int, left_pad: int) -> np.array:
        if top_pad > 0 and left_pad > 0:
            img = img[:, top_pad:, : -left_pad]
        elif top_pad > 0:
            img = img[:, top_pad:, :]
        elif left_pad > 0:
            img = img[:, :, :-left_pad]
        return img

    @staticmethod
    def _generate_output_img_path(dir_path: str, num: str,
                                  filename_format: str = "%06d_10",
                                  img_type: str = ".png"):
        return dir_path + filename_format % num + img_type

    @staticmethod
    def _depth2img(img: np.array) -> np.array:
        return (np.array(img) * float(DataSaver._DEPTH_DIVIDING)).astype(np.uint16)

    @staticmethod
    def _save_png_img(path: str, img: np.array) -> None:
        # save the png file
        cv2.imwrite(path, img)
