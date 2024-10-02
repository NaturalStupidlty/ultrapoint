import torch
import numpy
import torchvision.transforms as transforms

from loguru import logger
from ultrapoint.utils.loader import get_module


class DataLoadersFactory:
    @staticmethod
    def create(config: dict, dataset: str, mode: str):
        assert mode in ["train", "test"], f"Mode {mode} not supported"

        datasets_sources = "ultrapoint.datasets"
        dataset_module = get_module(dataset, datasets_sources)
        workers = config["data"].get(f"{mode}_workers", 1)
        batch_size = (
            config["model"]["batch_size"]
            if mode == "train"
            else config["model"]["eval_batch_size"]
        )

        logger.info(f"{mode.upper()} Workers: {workers}")
        logger.info(f"{mode.upper()} Dataset: {dataset_module.__name__}")

        dataset = dataset_module(
            transform=transforms.Compose([transforms.ToTensor()]),
            task=mode,
            **config["data"],
        )

        return torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=workers,
            worker_init_fn=DataLoadersFactory.initialize_worker,
        )

    @staticmethod
    def initialize_worker(worker_id: int):
        """
        The function is designed for pytorch multiprocess dataloader.
        Note that we use the pytorch random generator to generate a base_seed.
        Please try to be consistent.

        References:
           https://pytorch.org/docs/stable/notes/faq.html#dataloader-workers-random-seed

        """
        base_seed = torch.IntTensor(1).random_().item()
        numpy.random.seed(base_seed + worker_id)
