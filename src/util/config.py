# (c) 2021 Theo Costain
from typing import Union, List, Optional
from pathlib import Path
from enum import Enum, auto
from dataclasses import dataclass

from pytorch_lightning import LightningModule, LightningDataModule
from omegaconf import OmegaConf, MISSING
from torch.utils import data

from . import ExperimentConfig, Models

# from models.cnnx import CNN, CIFAR10DataModule
from models.cifar import CIFARModel
from models.imagenet import ImageNetModel


# General config
def load_config(path: Path, freeze: bool = True) -> ExperimentConfig:
    """Loads config file.

    Args:
        path (str): path to config file
    """
    # Load default config
    default_config: ExperimentConfig = OmegaConf.structured(ExperimentConfig)

    if path is None:
        return default_config

    # Load config from provided yaml file
    special_config = OmegaConf.load(path)

    # If special config gives an inheirit file load that and merge with base and special
    # else merge base and special
    if (inherit_path := special_config.inheirit_from) is not None:
        inherit_file = Path(inherit_path)
        assert inherit_file.exists() and inherit_file.is_file()
        inherit_config = load_config(inherit_file)

        final_config: ExperimentConfig = OmegaConf.merge(
            default_config, inherit_config, special_config
        )  # type:ignore # ducktyping
    else:
        final_config: ExperimentConfig = OmegaConf.merge(
            default_config, special_config
        )  # type:ignore # ducktyping

    if freeze:
        OmegaConf.set_readonly(final_config, True)  # type:ignore # ducktyping error

    return final_config


def get_model(config: ExperimentConfig) -> LightningModule:
    if config.model.type is Models.CIFAR10:
        return CIFARModel(
            operation=config.model.operator.value,
            lr=config.training.lr,
            milestones=OmegaConf.to_container(config.training.milestones),
            orders=OmegaConf.to_container(config.model.orders),
        )
    elif config.model.type is Models.IMAGENET:
        return ImageNetModel(
            operation=config.model.operator.value,
            orders=OmegaConf.to_container(
                config.model.orders
            ),  # bodge to avoid yaml.dump problems
            lr=config.training.lr,
            network=config.model.network,
        )
    else:
        raise ValueError("Dataset not supported")
