# (c) Theo Costain 2021
from itertools import product
from typing import List, Union
from argparse import ArgumentParser
from pathlib import Path
import logging
import ast

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning.callbacks import LearningRateMonitor
import pandas as pd
from tqdm import tqdm

from util.config import ExperimentConfig, load_config
from models.imagenet import ImageNetModel
from operations import CosineConv2d

from operations.aproxconv import regress_and_set_mode

logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)


def get_serch_orders(
    max_orders: List[int] = [9, 5, 5, 5, 5], min_orders: List[int] = [4, 2, 2, 2, 2]
) -> List[List[Union[int, List[int]]]]:
    assert len(max_orders) == 5
    u1, u2, u3, u4, u5 = max_orders
    l1, l2, l3, l4, l5 = min_orders

    return [
        [l1, [l2, l2], [l3, l3], [l4, l4], [l5, l5]]
        for l1, l2, l3, l4, l5 in product(
            range(l1, u1), range(l2, u2), range(l3, u3), range(l4, u4), range(l5, u5)
        )
    ]


def orders_to_params(orders: List[Union[int, List[int]]]) -> int:
    params = 0
    # conv1
    params += (orders[0] ** 2) * 64 * 3
    # layer 1
    params += (2 * orders[1][0] ** 2) * 64 * 64
    params += (2 * orders[1][1] ** 2) * 64 * 64
    # layer 2
    params += (orders[2][0] ** 2) * 128 * 64
    params += (orders[2][0] ** 2) * 128 * 128
    params += (2 * orders[2][1] ** 2) * 128 * 128
    # layer 3
    params += (orders[3][0] ** 2) * 128 * 256
    params += (orders[3][0] ** 2) * 256 * 256
    params += (2 * orders[3][1] ** 2) * 256 * 256
    # layer 4
    params += (orders[4][0] ** 2) * 512 * 256
    params += (orders[4][0] ** 2) * 256 * 256
    params += (2 * orders[4][1] ** 2) * 256 * 256
    return params


def test_configuration(cfg: ExperimentConfig, orders) -> pd.DataFrame:
    model = ImageNetModel(
        operation=cfg.model.operator.value,
        orders=orders,
        network=cfg.model.network,
        lr=cfg.training.lr
        # operation=CosineConv2d, orders=orders, lr=CONFIG.training.lr, workers=20
    ).cuda()
    seed_everything(42, workers=True)
    t_logger = TensorBoardLogger(
        cfg.logging.out_dir,
        name=cfg.name,
        default_hp_metric=False,
        version=str(orders),
    )

    trainer = Trainer(
        gpus=-1,
        logger=t_logger,
        accelerator="gpu",
        max_epochs=cfg.pl.max_epochs,
        enable_checkpointing=False,
        strategy=DDPPlugin(find_unused_parameters=True),
        callbacks=[LearningRateMonitor()]
        # enable_progress_bar=False,
    )
    results = pd.DataFrame(
        columns=[
            "config",
            "top1",
            "top5",
            "loss",
            "top1_post",
            "top5_post",
            "loss_post",
            "num_params",
        ]
    )

    model = model.apply(regress_and_set_mode)
    # assert model.model.conv1.operating_mode == "Approx"
    # assert model.model.layer1[0].conv1.operating_mode == "Approx"
    # assert model.model.layer2[0].conv1.operating_mode == "Approx"
    # assert model.model.layer3[0].conv1.operating_mode == "Approx"
    # assert model.model.layer4[0].conv1.operating_mode == "Approx"

    res = trainer.validate(model, verbose=False)
    res_dict = res[0]

    trainer.fit(model)

    res_post = trainer.validate(model, verbose=False)
    res_dict_post = res_post[0]
    results.loc[0] = [
        orders,
        res_dict["val/top1"],
        res_dict["val/top5"],
        res_dict["val/loss"],
        res_dict_post["val/top1"],
        res_dict_post["val/top5"],
        res_dict_post["val/loss"],
        orders_to_params(orders),
    ]
    ldir = Path(trainer.log_dir)
    trainer.save_checkpoint(ldir/"last.ckpt")

    with open(ldir / "results.txt", "w") as f:
        f.write(results.to_string())
    results.to_pickle(ldir / "results.md")
    print(results)

    return results


if __name__ == "__main__":

    # Parse commandline arguments
    PARSER = ArgumentParser("Train experiments")
    PARSER.add_argument(
        "--config_file",
        "-c",
        type=str,
        help="Experiment config file to merge with default",
    )
    PARSER.add_argument("--orders", "-o", type=str)
    ARGS = PARSER.parse_args()
    ord = ast.literal_eval(ARGS.orders)

    # Load and set configuration
    CONFIG = load_config(ARGS.config_file)
    test_configuration(CONFIG, ord)
