# (c) Theo Costain 2021
from argparse import ArgumentParser
from pathlib import Path
import subprocess

from pytorch_lightning import Trainer, seed_everything, Callback
from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from operations import Operations

from util.config import ExperimentConfig, get_model, load_config
from train import PWDCallback

from operations.aproxconv import regress_and_set_mode


def main(cfg: ExperimentConfig):

    t_logger = TensorBoardLogger(
        CONFIG.logging.out_dir, name=CONFIG.name, default_hp_metric=False
    )

    pwd_callback = PWDCallback()

    seed_everything(42, workers=True)

    trainer = Trainer(
        gpus=1,
        logger=t_logger,
        accelerator="gpu",
        callbacks=[pwd_callback],
    )

    model = get_model(cfg).cuda()

    print("Testing loaded pre-trained model")
    res = trainer.test(model, ckpt_path=CONFIG.pl.resume_path)
    print(res)
    with open(Path(trainer.log_dir) / "results.txt", "w") as f:
        f.write(str(res))

    if cfg.model.operator is Operations.CONV2D:
        return

    print("Switching to approximation mode and regressing approximate kernels")
    model.apply(regress_and_set_mode)

    print("Testing approximate kernels")
    res = trainer.test(model, ckpt_path=CONFIG.pl.resume_path)
    print(res)
    with open(Path(trainer.log_dir) / "results_approx.txt", "w") as f:
        f.write(str(res))


if __name__ == "__main__":

    # Parse commandline arguments
    PARSER = ArgumentParser("Train experiments")
    PARSER.add_argument(
        "--config_file",
        "-c",
        type=str,
        help="Experiment config file to merge with default",
    )
    ARGS = PARSER.parse_args()

    # Load and set configuration
    CONFIG = load_config(ARGS.config_file)

    main(CONFIG)
