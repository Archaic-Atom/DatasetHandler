# -*- coding: utf-8 -*-
import argparse
import torch
import cv2

try:
    from .dataset_reader import ChangeDetectionDataset
except ImportError:
    from dataset_reader import ChangeDetectionDataset


class ChangeDetectionDatasetUnitTest(object):
    def __init__(self):
        super().__init__()

    @ staticmethod
    def create_args() -> object:
        parser = argparse.ArgumentParser(
            description="The deep learning framework (based on pytorch)")
        parser.add_argument('--imgWidth', type=int, default=512, help='croped width')
        parser.add_argument('--imgHeight', type=int, default=512, help='croped height')
        parser.add_argument('--dataset', type=str, default='LEVIR-CD', help='dataset')
        parser.add_argument('--datasetPath', type=str,
                            default='./Datasets/levir_cd_training_list.csv',
                            help='the path of dataset')
        parser.add_argument('--resultImgDir', type=str, default='./Datasets/',
                            help='the path of dataset')
        return parser.parse_args()

    @ staticmethod
    def gen_input_data(args: object) -> tuple:
        dataset = ChangeDetectionDataset(args, args.datasetPath, True)
        training_dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=2, shuffle=False, num_workers=2,
            pin_memory=True, sampler=None)
        return training_dataloader

    @ staticmethod
    def gen_save_data(args: object) -> tuple:
        pass
        # saver = ReconstructionSaver(args)
        # return saver

    def exec(self, args: object) -> None:
        training_dataloader = self.gen_input_data(args)
        self.gen_save_data(args)
        for iteration, batch_data in enumerate(training_dataloader):
            print(iteration)
            print(batch_data[0].size())
            print(batch_data[1].size())
            print(batch_data[2].size())
            return


def main() -> None:
    unit_test = ChangeDetectionDatasetUnitTest()
    args = unit_test.create_args()
    unit_test.exec(args)


if __name__ == '__main__':
    main()
