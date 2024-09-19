# -*- coding: utf-8 -*-
import argparse
import torch

try:
    from .dataset_reader import StereoDataset
except ImportError:
    from dataset_reader import StereoDataset


class StereoDatasetUnitTest(object):
    def __init__(self):
        super().__init__()

    @staticmethod
    def create_args() -> object:
        parser = argparse.ArgumentParser(
            description="The deep learning framework (based on pytorch)")
        parser.add_argument('--imgWidth', type=int, default=512, help='croped width')
        parser.add_argument('--imgHeight', type=int, default=256, help='croped height')
        parser.add_argument('--dataset', type=str, default='rob', help='dataset')
        parser.add_argument('--dispNum', type=int, default=256, help='disp number')
        parser.add_argument('--startDisp', type=int, default=0, help='start disp')
        parser.add_argument('--datasetPath', type=str,
                            default='./Datasets/kitti2015_stereo_training_list.csv',
                            help='the path of dataset')
        return parser.parse_args()

    @staticmethod
    def gen_input_data(args: object) -> tuple:
        dataset = StereoDataset(args, args.datasetPath, True)
        training_dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=2, shuffle=False, num_workers=2,
            pin_memory=True, sampler=None)
        return training_dataloader

    def exec(self, args: object) -> None:
        training_dataloader = self.gen_input_data(args)
        for iteration, batch_data in enumerate(training_dataloader):
            print(iteration)
            print(batch_data[0].size())
            print(batch_data[1].size())
            print(batch_data[2].size())


def main():
    unit_test = StereoDatasetUnitTest()
    args = unit_test.create_args()
    unit_test.exec(args)


if __name__ == '__main__':
    main()
