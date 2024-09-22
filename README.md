# Custom Dataset Library for PyTorch and JackFramework

This is a PyTorch-based Dataset library designed to integrate with JackFramework, facilitating efficient data handling for machine learning and deep learning applications.

## Features

- **Seamless Integration with PyTorch**: Built on top of PyTorch's Dataset class, allowing for easy use in PyTorch-based projects.
- **JackFramework Compatibility**: Designed to work specifically with JackFramework, ensuring smooth workflow integration.
- **Customizable**: Allows users to easily modify data preprocessing, augmentation, and other transformations.

## Installation

1. To use this Dataset library, you need to have the following dependencies installed:

```bash
pip install torch
pip install jackframework
```

If JackFramework is not already installed, you can find it [JackFramework GitHub page](https://github.com/Archaic-Atom/JackFramework).

2. Install this library:

```bash
./build.sh
```

## Usage

```python
import DatasetHandler as dh

class StereoDataloader(jf.UserTemplate.DataHandlerTemplate):
    def get_train_dataset(self, path: str, is_training: bool = True) -> object:
        args = self.__args
        assert args.trainListPath == path
        self.__train_dataset = dh.StereoDataset(args, args.trainListPath, is_training)
        return self.__train_dataset

    def get_val_dataset(self, path: str) -> object:
        args = self.__args
        assert args.valListPath == path
        self.__val_dataset = dh.StereoDataset(args, args.valListPath, False)
        return self.__val_dataset
```