import random
from pathlib import Path
from typing import Optional, Type

import pytorch_lightning as pl
import torch
import torchvision
from torch.utils.data import DataLoader, random_split
from torchvision.datasets.mnist import MNIST

__all__ = ["MNISTDataModule", "GenericTVDataModule"]


class MNISTDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: Path = Path("dataset/mnist-dataset"), batch_size: int = 256, download: bool = True):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.download = download
        self.transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                # torchvision.transforms.Normalize((0.1307,), (0.3081,))
            ]
        )

    def setup(self, stage: Optional[str] = None):
        self.mnist_train_val = MNIST(
            self.data_dir / "train", download=self.download, train=True, transform=self.transform
        )
        self.mnist_test = MNIST(self.data_dir / "test", download=self.download, train=False, transform=self.transform)
        self.mnist_val, self.mnist_train = random_split(
            self.mnist_train_val, [10000, 50000], generator=torch.Generator().manual_seed(42)
        )

    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=self.batch_size)

    def predict_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=self.batch_size)

    def teardown(self, stage: Optional[str] = None):
        pass


class ResizeGrayfy:
    def __init__(self, min_size, max_size=None):
        self.min_size = min_size
        if max_size is None:
            max_size = min_size
        self.max_size = max_size

    def __call__(self, image):
        size = random.randint(self.min_size, self.max_size)
        image = torchvision.transforms.functional.pil_to_tensor(image)
        image = torchvision.transforms.functional.resize(image, (size, size))
        image = torchvision.transforms.functional.rgb_to_grayscale(image)
        return image


class GenericTVDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_class: Type,
        data_dir: Path = Path("dataset"),
        batch_size: int = 256,
        download: bool = True,
        includes_val: bool = False,
        use_resize: bool = True,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.download = download
        self._data_class = data_class
        self._has_splits = includes_val
        if use_resize:
            self.transform = ResizeGrayfy(28)
        else:
            self.transform = torchvision.transforms.Compose(
                [
                    torchvision.transforms.ToTensor(),
                    # torchvision.transforms.Normalize((0.1307,), (0.3081,))
                ]
            )

    def _split_configure_dataloaders(self):
        self.data_train_val = self._data_class(
            self.data_dir, download=self.download, train=True, transform=self.transform
        )
        self.data_test = self._data_class(self.data_dir, download=self.download, train=False, transform=self.transform)

        size = len(self.data_train_val)
        train_size = int(size * 0.9)
        self.data_val, self.data_train = random_split(
            self.data_train_val, [size - train_size, train_size], generator=torch.Generator().manual_seed(42)
        )

    def _configure_std_dataloaders(self):
        self.data_train = self._data_class(
            self.data_dir, download=self.download, split="train", transform=self.transform
        )
        try:
            self.data_val = self._data_class(
                self.data_dir, download=self.download, split="val", transform=self.transform
            )
        except:
            self.data_val = self._data_class(
                self.data_dir, download=self.download, split="valid", transform=self.transform
            )

        self.data_test = self._data_class(self.data_dir, download=self.download, split="test", transform=self.transform)

    def setup(self, stage: Optional[str] = None):
        if not self._has_splits:
            self._split_configure_dataloaders()
        else:
            self._configure_std_dataloaders()

    def train_dataloader(self):
        return DataLoader(self.data_train, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.data_val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.data_test, batch_size=self.batch_size)

    def predict_dataloader(self):
        return DataLoader(self.data_test, batch_size=self.batch_size)

    def teardown(self, stage: Optional[str] = None):
        pass
