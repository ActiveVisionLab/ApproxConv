from typing import List, Optional, Tuple
from pathlib import Path

import torch
from torch import nn
from pytorch_lightning import LightningModule
from torchmetrics import Accuracy
from torchvision.datasets import CIFAR10
from torch.utils.data.dataloader import DataLoader, default_collate
from torch.optim import SGD
from torch.optim.lr_scheduler import MultiStepLR
from torchvision import transforms
from tqdm import tqdm

from models.resnet import resnet18
from models.resnet_cifar import ResNet18, QUANTIZED_ResNet18
from models.resnet_s import resnet20, resnet32

# from models.convnext import convnext_tiny


NETWORKS = {
    "resnet18": resnet18,
    "resnet18_cifar": ResNet18,
    "q_resnet_18_cifar": QUANTIZED_ResNet18,
    "resnet20": resnet20,
    "resnet32": resnet32,
}


class CIFARModel(LightningModule):
    def __init__(
        self,
        batch_size: int = 128,
        workers: int = 8,
        lr: float = 1e-1,
        operation=nn.Conv2d,
        orders=None,
        network="resnet20",
        milestones: List[int] = [5],
        bit: Optional[List[int]] = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        model = NETWORKS[network]

        if operation == nn.Conv2d:
            self.model = model(pretrained=True, strict=True)
        elif network == "q_resnet_18_cifar":
            self.model = model(
                pretrained=True,
                strict=False,
                operation=operation,
                orders=orders,
                bits=bit,
            )
        else:
            self.model = model(
                pretrained=True,
                strict=False,
                operation=operation,
                orders=orders,
            )

        self.criterion = nn.CrossEntropyLoss()
        self.top1 = Accuracy()
        self.top5 = Accuracy(top_k=5)

    def training_step(
        self, data: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:

        inputImage, target = data

        outputs = self.model(inputImage)

        loss = self.criterion(outputs, target)

        self.log("train/loss", loss)
        return loss

    def validation_step(
        self, data: Tuple[torch.Tensor, torch.Tensor], batch_idx: int, tag: str = "val"
    ) -> torch.Tensor:
        inputImage, target = data

        outputs = self.model(inputImage)

        val_loss = self.criterion(outputs, target)

        self.log(f"{tag}/top1", self.top1(outputs, target))
        self.log(f"{tag}/top5", self.top5(outputs, target))

        self.log(f"{tag}/loss", val_loss)
        return val_loss

    def test_step(
        self, data: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        return self.validation_step(data, batch_idx, tag="test")

    def configure_optimizers(self):
        optim = SGD(
            self.parameters(), lr=self.hparams.lr, momentum=0.9, weight_decay=5e-4
        )
        sched = MultiStepLR(optimizer=optim, milestones=self.hparams.milestones)
        return [optim], [sched]

    def train_dataloader(self) -> DataLoader:
        if self.hparams.network in ["resnet20", "resnet32"]:
            norm_tup = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        else:
            norm_tup = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        train_transforms = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(*norm_tup),
            ]
        )
        fast = Path("../data/cifar10_fast")
        if fast.exists():
            path = fast
        else:
            path = Path("../data/cifar10")
        cifar_train = CIFAR10(
            root=path,
            train=True,
            download=False,
            transform=train_transforms,
        )

        return DataLoader(
            cifar_train,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=self.hparams.workers,
        )

    def val_dataloader(self) -> DataLoader:
        if self.hparams.network in ["resnet20", "resnet32"]:
            norm_tup = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        else:
            norm_tup = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        val_transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(*norm_tup),
            ]
        )
        fast = Path("../data/cifar10_fast")
        if fast.exists():
            path = fast
        else:
            path = Path("../data/cifar10")
        cifar_val = CIFAR10(
            root=path,
            train=False,
            download=False,
            transform=val_transforms,
        )

        return DataLoader(
            cifar_val,
            batch_size=self.hparams.batch_size,
            pin_memory=True,
            num_workers=self.hparams.workers,
        )
