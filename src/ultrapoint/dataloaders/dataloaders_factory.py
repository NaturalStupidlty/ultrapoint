import torch
import numpy
import torchvision.transforms as transforms
from torch.utils.data import DataLoader as DataLoader

from loguru import logger
from ultrapoint.datasets.base_images import ImagesDataset
from ultrapoint.datasets.synthetic_dataset_gaussian import SyntheticDatasetGaussian


class DataLoadersFactory:
    SUPPORTED_DATASETS = {
        "ImagesDataset": ImagesDataset,
        "SyntheticDatasetGaussian": SyntheticDatasetGaussian,
    }

    @staticmethod
    def create(config: dict, dataset_name: str, mode: str):
        assert mode in ["train", "test"], f"Mode {mode} not supported"
        assert (
            dataset_name in DataLoadersFactory.SUPPORTED_DATASETS
        ), f"Dataset {dataset_name} not supported"

        workers = config["data"].get(f"{mode}_workers", 1)
        batch_size = (
            config["model"]["batch_size"]
            if mode == "train"
            else config["model"]["eval_batch_size"]
        )

        logger.info(f"{mode.upper()} Workers: {workers}")
        logger.info(f"{mode.upper()} Dataset: {dataset_name}")

        dataset_module = DataLoadersFactory.SUPPORTED_DATASETS[dataset_name]
        dataset = dataset_module(
            transforms=transforms.Compose([transforms.ToTensor()]),
            task=mode,
            **config["data"],
        )

        return DataLoader(
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
