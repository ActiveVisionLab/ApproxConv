"""
(c) Marcelo Genanri 2019 & Theo Costain 2021
Implementation of cnnx (test cnn) to train Block Floating Point (BFP) and DSConv
using CIFAR10 dataset.
Adapeted for use in testing FlexRes
"""
from typing import List, Optional, Tuple, Any
import torch
import torch.nn.functional as F
from torch import nn
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning.core.datamodule import LightningDataModule
from torch.nn import Module
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR
from torchvision.datasets import CIFAR10
from torchvision.transforms import transforms
from torch.utils.data import DataLoader

from operations import operator


class BaseConv(nn.Module):
    def __init__(
        self, operator: Module, in_planes: int, out_planes: int, kernel: int, **kwargs
    ):
        super(BaseConv, self).__init__()
        self.conv = operator(in_planes, out_planes, kernel, **kwargs)
        self.bn = nn.BatchNorm2d(out_planes)

    def forward(self, x: torch.Tensor):
        return F.relu(self.bn(self.conv(x)))


class CNN(LightningModule):
    """
    Module to test FlexConv module on CIFAR10
    """

    def __init__(
        self,
        operator: Operators = Operators.CONV2D,
        lr: float = 0.01,
        milestones: List[int] = [200, 300, 400],
    ):
        super().__init__()
        self.operator = operator.value
        self.save_hyperparameters()

        self.conv1 = self.operator(3, 64, (3, 3), padding=1)
        self.bn1 = nn.BatchNorm2d(64)

        self.features1, outch = self.__make_layers__(64, 3)
        self.max_pool1 = nn.MaxPool2d(2, stride=2)

        self.features2, outch = self.__make_layers__(outch, 3)
        self.max_pool2 = nn.MaxPool2d(2, stride=2)

        self.features3, outch = self.__make_layers__(outch, 3)
        self.avg_pool = nn.AvgPool2d(8)

        self.linear = nn.Linear(outch, 10)

        self.loss_fn = nn.CrossEntropyLoss()

    def __make_layers__(
        self, initial_channel: int, expansion: int
    ) -> "tuple[nn.Module, int]":
        number_layers = 3
        layers = []
        for i in range(number_layers):
            inch = initial_channel * (2 ** (i // expansion))
            outch = initial_channel * (2 ** ((i + 1) // expansion))
            layers.append(BaseConv(self.operator, inch, outch, (3, 3), padding=1))

        return nn.Sequential(*layers), outch

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.features1(x)
        x = self.max_pool1(x)
        x = self.features2(x)
        x = self.max_pool2(x)
        x = self.features3(x)
        x = self.avg_pool(x)

        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x

    def configure_optimizers(self):
        optimiser = Adam(self.parameters(), lr=self.hparams.lr)
        lr_sched = MultiStepLR(optimiser, milestones=self.hparams.milestones)
        return {"optimizer": optimiser, "lr_scheduler": lr_sched}

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        images, labels = batch
        logits = self(images)
        loss = self.loss_fn(logits, labels)

        self.log("train/loss", loss)

        return loss

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        images, labels = batch
        logits = self(images)

        loss = self.loss_fn(logits, labels)
        predicted = torch.argmax(logits, dim=1)
        correct = (predicted == labels).sum()
        total = labels.shape[0]

        return {"loss": loss, "correct": correct, "total": total}

    def validation_epoch_end(self, outputs: List[Any]) -> None:
        val_loss = torch.stack([x["loss"] for x in outputs]).mean()
        correct = torch.stack([x["correct"] for x in outputs]).sum().float()
        total = sum([x["total"] for x in outputs])
        val_acc = correct / total
        self.log("val/loss", val_loss)
        self.log("val/acc", val_acc)
        return


class CIFAR10DataModule(LightningDataModule):
    def __init__(self, batch_size: int, num_workers: int) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_transforms = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        )

        self.val_transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        )

    def setup(self, stage: Optional[str] = None):
        self.cifar_train = CIFAR10(
            root="../data/cifar10",
            train=True,
            download=False,
            transform=self.train_transforms,
        )
        self.cifar_val = CIFAR10(
            root="../data/cifar10",
            train=False,
            download=False,
            transform=self.val_transforms,
        )

    def prepare_data(self):
        CIFAR10(
            root="../data/cifar10",
            train=True,
            download=True,
        )
        CIFAR10(
            root="../data/cifar10",
            train=False,
            download=True,
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.cifar_train,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=torch.cuda.is_available(),
            num_workers=self.num_workers,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.cifar_val,
            batch_size=self.batch_size,
            pin_memory=torch.cuda.is_available(),
            num_workers=self.num_workers,
        )
