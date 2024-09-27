import os
import time
import sys
from loguru import logger
from torch.utils.data import DataLoader


def create_logger(level: str, logs_dir: str) -> None:
    os.makedirs(logs_dir, exist_ok=True)
    logger.remove()
    filename = f"{time.strftime('%Y-%m-%d_%H-%M-%S')}.log"
    logger.add(os.path.join(logs_dir, filename), level="DEBUG", mode="w")
    logger.add(sys.stdout, level=level)


def log_data_size(train_loader: DataLoader, config: dict, tag: str = "train") -> None:
    logger.info(
        "== %s split size %d in %d batches"
        % (tag, len(train_loader) * config["model"]["batch_size"], len(train_loader))
    )
