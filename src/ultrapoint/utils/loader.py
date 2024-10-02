"""many loaders
# loader for model, dataset, testing dataset
"""
import os
import re
import importlib
import torch
import torch.optim

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
