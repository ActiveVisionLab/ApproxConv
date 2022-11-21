# (c) Marcelo Genanri 2019 & Theo Costain 2020
import logging
import subprocess
import sys
import traceback
from datetime import datetime
from pathlib import Path
from shutil import rmtree

from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
import torch
from tqdm import tqdm

from models.resnet import resnet18
from modules.aproxconv import set_operating_mode

CONFIG_DICT = {"pretrained": True}


def train(model, writer, device, epochs=1, lr=1e-6, batch_size=256):
    preprocess = transforms.Compose(
        [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    logging.info("Loading data")
    data_loader = torch.utils.data.DataLoader(
        ImageFolder("../data/ILSVRC2012/ILSVRC2012_train", preprocess),
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
    )
    logging.info("Done loading data")

    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)

    model = model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for i, (inputImage, target) in enumerate(tqdm((data_loader))):
            step = i + epoch * len(data_loader)

            optimizer.zero_grad()

            target = target.to(device)
            inputImage = inputImage.to(device)

            output = model(inputImage)

            loss = criterion(output, target)
            loss.backward()

            optimizer.step()

            writer.add_scalar("train/loss", loss, step)
            writer.add_histogram("train/activ/out", output, step)

            running_loss += loss.item()


def test(model, device, batch_size=896):
    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )

    logging.info("Loading data")
    data_loader = torch.utils.data.DataLoader(
        ImageFolder("../data/ILSVRC2012/ILSVRC2012_val", transform),
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
    )
    logging.info("Done loading data")

    model.eval()
    correct1, correct5, total = 0, 0, 0
    with torch.no_grad():
        for i, (inputImage, target) in enumerate(tqdm(data_loader)):
            target = target.to(device)
            inputImage = inputImage.to(device)

            outputs = model(inputImage)
            _, predicted = torch.max(outputs.data, 1)
            a = torch.argsort(outputs.data, 1, True)[:, 0:5]

            total += target.size(0)
            correct1 += (predicted == target).sum().item()
            correct5 += (a == target.unsqueeze(1)).sum().item()

    logging.info(f"Accuracy1: {correct1 / total}")
    logging.info(f"Accuracy5: {correct5 / total}")


def main(LOG_DIR):
    logging.info(CONFIG_DICT)
    if CONFIG_DICT["pretrained"]:
        logging.info("Loading pretrained model")
        model = resnet18(pretrained=True, strict=False)
    else:
        logging.info("Loading clean model")
        model = resnet18(pretrained=False, strict=False)

    if torch.cuda.device_count() >= 1:
        model = torch.nn.DataParallel(model)
        device = torch.device("cuda:0" if torch.cuda.is_available else "cpu")
    else:
        device = "cpu"
    logging.debug(f"Using device {device}")
    model = model.to(device)

    writer = SummaryWriter(LOG_DIR)

    logging.info("Testing original (pretrained) model")
    test(model, device)

    logging.info("Switching operating mode (not regressing parameters)")
    model.apply(set_operating_mode)

    logging.info("Testing converted model")
    test(model, device)

    logging.info("Training converted model")
    train(model, writer, device, epochs=2)

    torch.save(model.module.state_dict(), LOG_DIR / "imagenet.pth")

    logging.info("Testing converted model")
    test(model, device)


if __name__ == "__main__":
    LOG_DIR = Path("../log/imagenet")
    LOG_DIR = LOG_DIR / datetime.now().strftime("%Y-%b-%d-%H-%M")
    if not LOG_DIR.is_dir():
        LOG_DIR.mkdir()
    logging.basicConfig(
        format="[{asctime}][{levelname}] {message}",
        style="{",
        handlers=[
            logging.FileHandler(filename=LOG_DIR / "train_log.txt", mode="w"),
            logging.StreamHandler(sys.stdout),
        ],
        level=20,
    )
    git_id = (
        subprocess.run(["git", "rev-parse", "--short", "HEAD"], stdout=subprocess.PIPE)
        .stdout.decode("utf-8")
        .strip()
    )
    git_diff = subprocess.run(["git", "diff", "--color"], stdout=subprocess.PIPE).stdout
    with open(LOG_DIR / "diff.diff", "wb") as f:
        f.write(git_diff)
    logging.info(f"Git HEAD is {git_id}")

    try:
        main(LOG_DIR)
    except (KeyboardInterrupt, Exception) as e:
        logging.error(e)
        traceback.print_exc()
        res = input("Delete logdir and contents? (y/n): ")
        if res[0] == "y":
            logging.warning(f"Deleting logdir {LOG_DIR}")
            rmtree(LOG_DIR)
