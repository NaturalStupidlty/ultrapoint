"""many loaders
# loader for model, dataset, testing dataset
"""

import os
import re
import importlib
import numpy as np
import torch
import torch.optim
import torchvision.transforms as transforms

from loguru import logger
from src.ultrapoint.utils.utils import load_checkpoint


def get_checkpoints_path(output_dir):
    """
    This func
    :param output_dir:
    :return:
    """
    save_path = os.path.join(output_dir, 'checkpoints')
    logger.info(f"Saving everything to {save_path}")
    os.makedirs(save_path, exist_ok=True)
    return save_path


def get_module(name, path: str = "") -> callable:
    path = camel_to_snake(name) if not path else f"{path}.{camel_to_snake(name)}"
    module = importlib.import_module(path)
    return getattr(module, name)


class DataLoadersFabric:
    @staticmethod
    def create(config: dict, dataset: str, mode: str):
        assert mode in ['train', 'test'], f"Mode {mode} not supported"

        datasets_sources = "ultrapoint.datasets"
        dataset_module = get_module(dataset, datasets_sources)
        workers = config["data"].get(f"{mode}_workers", 1)
        batch_size = config["model"]["batch_size"] if mode == 'train' else config["model"]["eval_batch_size"]

        logger.info(f"{mode.upper()} Workers: {workers}")
        logger.info(f"{mode.upper()} Dataset: {dataset_module.__name__}")

        dataset = dataset_module(
            transform=transforms.Compose([transforms.ToTensor()]),
            task=mode,
            **config['data']
        )

        return torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=workers,
            worker_init_fn=DataLoadersFabric.initialize_worker,
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
        np.random.seed(base_seed + worker_id)


def worker_init_fn(worker_id):
    """The function is designed for pytorch multi-process dataloader.
    Note that we use the pytorch random generator to generate a base_seed.
    Please try to be consistent.

    References:
       https://pytorch.org/docs/stable/notes/faq.html#dataloader-workers-random-seed

    """
    base_seed = torch.IntTensor(1).random_().item()
    np.random.seed(base_seed + worker_id)


def camel_to_snake(name: str) -> str:
    # Add underscores between words and convert to lowercase
    snake_case = re.sub(r'(?<!^)(?=[A-Z])', '_', name).lower()
    return snake_case


def get_model(name):
    mod = __import__('models.{}'.format(name), fromlist=[''])
    return getattr(mod, name)


def modelLoader(model='SuperPointNet', **options):
    logger.info(f"Creating model: {model}", )
    net = get_model(model)
    net = net(**options)
    return net


# mode: 'full' means the formats include the optimizer and epoch
# full_path: if not full path, we need to go through another helper function
def pretrainedLoader(net, optimizer, epoch, path, mode='full', full_path=False):
    # load checkpoint
    if full_path == True:
        checkpoint = torch.load(path)
    else:
        checkpoint = load_checkpoint(path)
    # apply checkpoint
    if mode == 'full':
        net.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        #         epoch = checkpoint['epoch']
        epoch = checkpoint['n_iter']
    #         epoch = 0
    else:
        net.load_state_dict(checkpoint)
        # net.load_state_dict(torch.load(path,map_location=lambda storage, loc: storage))
    return net, optimizer, epoch
