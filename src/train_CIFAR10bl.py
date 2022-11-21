# (c) Theo Costain 2022
from argparse import ArgumentParser
from pathlib import Path
import subprocess

import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

from util.config import ExperimentConfig, load_config
from models.cifar import CIFARModel


def main():

    t_logger = TensorBoardLogger(
        "log/baseline/cifar10", name="CIFAR10-Baseline", default_hp_metric=False
    )
    chkpt_callback = ModelCheckpoint(
        monitor="val/top1",
        mode="max",
        save_last=True,
        save_top_k=3,
        every_n_epochs=1,
    )

    # seed_everything(42, workers=True)
    seed_everything(42069, workers=True)

    trainer = Trainer(
        gpus=1,
        logger=t_logger,
        accelerator="gpu",
        max_epochs=5,
        # max_epochs=350,
        callbacks=[chkpt_callback, LearningRateMonitor()],
        check_val_every_n_epoch=1,
        # check_val_every_n_epoch=5,
    )

    # model = CIFARModel(milestones=[150, 250], lr=0.001, workers=10, network="resnet18_cifar")
    model = CIFARModel(milestones=[3], lr=0.001, workers=10, network="resnet18_cifar")

    # trainer.fit(model)
    model.model.load_state_dict(torch.load('../data/resnet18.pth'))
    trainer.validate(model)

    trainer.fit(model)


if __name__ == "__main__":

    main()