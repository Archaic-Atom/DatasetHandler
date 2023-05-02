# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
import argparse
import torch

try:
    from .dataset_reader import ReconstructionDataset
    from .dataset_saver import ReconstructionSaver
except ImportError:
    from dataset_reader import ReconstructionDataset
    from dataset_saver import ReconstructionSaver


class ReconstructionDatasetUnitTest(object):
    def __init__(self):
        super().__init__()

    @staticmethod
    def create_args() -> object:
        parser = argparse.ArgumentParser(
            description="The deep learning framework (based on pytorch)")
        parser.add_argument('--imgWidth', type=int, default=1024, help='croped width')
        parser.add_argument('--imgHeight', type=int, default=1024, help='croped height')
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

    @staticmethod
    def gen_input_data(args: object) -> tuple:
        dataset = ReconstructionDataset(args, args.datasetPath, False)
        training_dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=1, shuffle=False, num_workers=2,
            pin_memory=True, sampler=None)
        return training_dataloader

    @staticmethod
    def gen_save_data(args: object) -> tuple:
        saver = ReconstructionSaver(args)
        return saver

    def exec(self, args: object) -> None:
        training_dataloader = self.gen_input_data(args)
        saver = self.gen_save_data(args)
        for iteration, batch_data in enumerate(training_dataloader):
            print(iteration)
            print(batch_data[0].size())

            saver.save_output(batch_data[0].cpu().detach().numpy(), 0, 'whu',
                              [batch_data[1], batch_data[2],
                              batch_data[3], batch_data[4]], 0)


def main():
    unit_test = ReconstructionDatasetUnitTest()
    args = unit_test.create_args()
    unit_test.exec(args)


if __name__ == '__main__':
    main()
