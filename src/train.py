# (c) Theo Costain 2021
from argparse import ArgumentParser
from pathlib import Path
import subprocess

from pytorch_lightning import Trainer, seed_everything, Callback
from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

from util.config import ExperimentConfig, get_model, load_config

from operations.aproxconv import set_operating_mode


class GitDiffCallback(Callback):
    def on_fit_start(self, trainer, pl_module):
        self.save_git_diff(trainer)

    def on_test_start(self, trainer, pl_module):
        self.save_git_diff(trainer, "test")

    def save_git_diff(self, trainer, tag="fit"):
        git_id = (
            subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"], stdout=subprocess.PIPE
            )
            .stdout.decode("utf-8")
            .strip()
        )
        git_diff = subprocess.run(
            ["git", "diff", "--no-color"], stdout=subprocess.PIPE
        ).stdout

        diff_file = Path(trainer.log_dir) / f"{git_id}_{tag}.diff"
        if not diff_file.exists():
            diff_file.parent.mkdir(parents=True, exist_ok=True)
        with open(diff_file, "wb") as f:
            f.write(git_diff)


class PWDCallback(Callback):
    def on_fit_start(self, trainer, pl_module):
        pl_module.print(f"Current logging directory is: {trainer.log_dir}")

    def on_test_start(self, trainer, pl_module) -> None:
        pl_module.print(f"Current logging directory is: {trainer.log_dir}")


def main(cfg: ExperimentConfig):

    t_logger = TensorBoardLogger(
        CONFIG.logging.out_dir, name=CONFIG.name, default_hp_metric=False
    )
    chkpt_callback = ModelCheckpoint(
        monitor=CONFIG.logging.selection_metric,
        mode=CONFIG.logging.selection_mode,
        save_last=True,
        save_top_k=3,
        every_n_epochs=CONFIG.logging.n_epochs,
    )

    git_callback = GitDiffCallback()
    pwd_callback = PWDCallback()
    lr_callback = LearningRateMonitor(logging_interval="epoch")

    seed_everything(42, workers=True)
    # seed_everything(69, workers=True)

    trainer = Trainer(
        gpus=-1,
        logger=t_logger,
        max_epochs=CONFIG.pl.max_epochs,
        max_steps=CONFIG.pl.max_steps,
        accelerator="gpu",
        strategy=DDPPlugin(find_unused_parameters=True),
        callbacks=[git_callback, chkpt_callback, pwd_callback, lr_callback],
        overfit_batches=CONFIG.pl.overfit,
        check_val_every_n_epoch=CONFIG.pl.val_every,
        log_every_n_steps=CONFIG.logging.log_every,
    )

    model = get_model(cfg)
    model.apply(set_operating_mode)
    trainer.fit(model, ckpt_path=CONFIG.pl.resume_path)


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
