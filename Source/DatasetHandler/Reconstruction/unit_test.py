# -*- coding: utf-8 -*-
import argparse
import torch
import cv2

try:
    from .dataset_reader import ReconstructionDataset
    from .dataset_saver import ReconstructionSaver
except ImportError:
    from dataset_reader import ReconstructionDataset
    from dataset_saver import ReconstructionSaver


def reconstruct_img(
        img_size: tuple, mask_img_patch: torch.Tensor, random_sample_list: torch.Tensor) -> torch.Tensor:
    b, c, h, w = img_size
    _, p_n, _, block_h, block_w = mask_img_patch.shape
    _, r_n = random_sample_list.shape
    new_img = torch.zeros(b, c, h, w).to(mask_img_patch.device)
    assert p_n == r_n
    for i in range(b):
        for j in range(p_n):
            height_id = torch.div(random_sample_list[i, j], int(h / block_w), rounding_mode='floor')
            width_id = random_sample_list[i, j] % int(w / block_h)
            new_img[i, :, height_id * block_h:height_id * block_h + block_h,
                    width_id * block_w: width_id * block_w + block_w] =\
                mask_img_patch[i, j, :, :, :]
    return new_img


class ReconstructionDatasetUnitTest(object):
    def __init__(self):
        super().__init__()

    @ staticmethod
    def create_args() -> object:
        parser = argparse.ArgumentParser(
            description="The deep learning framework (based on pytorch)")
        parser.add_argument('--imgWidth', type=int, default=512, help='croped width')
        parser.add_argument('--imgHeight', type=int, default=512, help='croped height')
        parser.add_argument('--dataset', type=str, default='whu', help='dataset')
        parser.add_argument('--dispNum', type=int, default=256, help='disp number')
        parser.add_argument('--startDisp', type=int, default=0, help='start disp')
        parser.add_argument('--block_size', type=int, default=32, help='start disp')
        parser.add_argument('--mask_ratio', type=float, default=0.65, help='start disp')
        parser.add_argument('--datasetPath', type=str,
                            default='./Datasets/kitti2015_stereo_training_list.csv',
                            help='the path of dataset')
        parser.add_argument('--resultImgDir', type=str,
                            default='./Datasets/',
                            help='the path of dataset')
        return parser.parse_args()

    @ staticmethod
    def gen_input_data(args: object) -> tuple:
        dataset = ReconstructionDataset(args, args.datasetPath, True)
        training_dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=1, shuffle=False, num_workers=2,
            pin_memory=True, sampler=None)
        return training_dataloader

    @ staticmethod
    def gen_save_data(args: object) -> tuple:
        saver = ReconstructionSaver(args)
        return saver

    def exec(self, args: object) -> None:
        training_dataloader = self.gen_input_data(args)
        saver = self.gen_save_data(args)
        for iteration, batch_data in enumerate(training_dataloader):
            print(iteration)

            a = batch_data[0:3]

            print(a[0].size())
            print(a[1].size())
            print(a[2].size())

            new_img = reconstruct_img(batch_data[0].shape, batch_data[1], batch_data[2])
            print(new_img.shape)
            cv2.imwrite(args.resultImgDir + '0.png', new_img.cpu().detach().numpy()[0, 0, :, :] * 255)
            return

            saver.save_output(new_img.cpu().detach().numpy(), 0, 'whu',
                              [batch_data[3], batch_data[4],
                              batch_data[5], batch_data[6]], 0)


def main() -> None:
    unit_test = ReconstructionDatasetUnitTest()
    args = unit_test.create_args()
    unit_test.exec(args)


if __name__ == '__main__':
    main()
