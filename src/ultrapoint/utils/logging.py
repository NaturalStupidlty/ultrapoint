import os
import time
import sys
from loguru import logger
from torch.utils.data import DataLoader


def create_logger(level: str, directory: str) -> None:
    os.makedirs(directory, exist_ok=True)
    logger.remove()
    filename = f"{time.strftime('%Y-%m-%d_%H-%M-%S')}.log"
    logger.add(os.path.join(directory, filename), level="DEBUG", mode="w")
    logger.add(sys.stdout, level=level)


def log_data_size(train_loader: DataLoader, config: dict, tag: str = "train") -> None:
    logger.info(
        "%s split size %d in %d batches"
        % (tag, len(train_loader) * config["model"]["batch_size"], len(train_loader))
    )


def log_dict_attr(dictionary, attr=None):
    for item in list(dictionary):
        d = dictionary[item]
        if attr is None:
            logger.info(f"{item}: {d}")
        else:
            if hasattr(d, attr):
                logger.info(f"{item}: {getattr(d, attr)}")
            else:
                logger.info(f"{item}: {len(d)}")
