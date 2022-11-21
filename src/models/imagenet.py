# (c) Theo Costain 2021
from pathlib import Path
from typing import Tuple

import torch
from torch import nn
from pytorch_lightning import LightningModule
from torchmetrics import Accuracy
from torchvision.datasets import ImageFolder
from torch.utils.data.dataloader import DataLoader, default_collate
from torch.optim import SGD
from torch.optim.lr_scheduler import MultiStepLR
from torchvision import transforms
from tqdm import tqdm

from models.resnet import quant_resnet18, resnet18
from models.convnext import convnext_tiny


NETWORKS = {
    "resnet18": resnet18,
    "convnext_tiny": convnext_tiny,
    "q_resnet18": quant_resnet18,
}


class ImageNetModel(LightningModule):
    def __init__(
        self,
        batch_size=256,
        workers=8,
        lr=1e-4,
        operation=nn.Conv2d,
        orders=None,
        network="resnet18",
        pretrained=True,
        bit=None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        model = NETWORKS[network]

        if operation == nn.Conv2d:
            self.model = model(pretrained=pretrained)
        else:
            self.model = model(
                pretrained=pretrained,
                strict=False,
                operation=operation,
                orders=orders,
                bits=bit,
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
        sched = MultiStepLR(optimizer=optim, milestones=[3, 7, 12])
        return [optim], [sched]

    def train_dataloader(self) -> DataLoader:
        preprocess = transforms.Compose(
            [
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        return DataLoader(
            ImageFolder("../data/ILSVRC2012/ILSVRC2012_train", preprocess),
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.hparams.workers,
            pin_memory=True,
            collate_fn=default_collate,
        )

    def val_dataloader(self) -> DataLoader:
        transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )

        fast = Path("../data/ILSVRC2012/ILSVRC2012_val_shm")
        if fast.exists():
            path = str(fast)
        else:
            path = "../data/ILSVRC2012/ILSVRC2012_val"
        return DataLoader(
            ImageFolder(path, transform),
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.workers,
            pin_memory=True,
            collate_fn=default_collate,
        )

    def test_dataloader(self) -> DataLoader:
        return self.val_dataloader()
