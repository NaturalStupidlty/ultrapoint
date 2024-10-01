"""many loaders
# loader for model, dataset, testing dataset
"""

import os
import re
import logging
import importlib
import numpy as np
import torch
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms

from loguru import logger
from utils.utils import load_checkpoint


def get_checkpoints_path(output_dir):
    """
    This func
    :param output_dir:
    :return:
    """
    save_path = os.path.join(output_dir, 'checkpoints')
    logging.info('=> will save everything to {}'.format(save_path))
    os.makedirs(save_path, exist_ok=True)
    return save_path


def worker_init_fn(worker_id):
    """The function is designed for pytorch multi-process dataloader.
    Note that we use the pytorch random generator to generate a base_seed.
    Please try to be consistent.

    References:
       https://pytorch.org/docs/stable/notes/faq.html#dataloader-workers-random-seed

    """
    base_seed = torch.IntTensor(1).random_().item()
    np.random.seed(base_seed + worker_id)


def DataLoader(config, dataset):
    workers_train = config["data"].get('workers_train', 1)
    workers_val = config["data"].get('workers_val', 1)

    logging.info(f"Workers_train: {workers_train}, workers_val: {workers_val}")

    data_transforms = {
        'train': transforms.Compose([
            transforms.ToTensor(),
        ]),
        'val': transforms.Compose([
            transforms.ToTensor(),
        ]),
    }
    dataset = get_module(dataset, 'datasets')
    logger.info(f"Dataset: {dataset}")

    train_set = dataset(
        transform=data_transforms['train'],
        task='train',
        **config['data'],
    )
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=config['model']['batch_size'], shuffle=True,
        pin_memory=True,
        num_workers=workers_train,
        worker_init_fn=worker_init_fn
    )
    val_set = dataset(
        transform=data_transforms['train'],
        task='val',
        **config['data'],
    )
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=config['model']['eval_batch_size'], shuffle=True,
        pin_memory=True,
        num_workers=workers_val,
        worker_init_fn=worker_init_fn
    )

    return {'train_loader': train_loader, 'val_loader': val_loader,
            'train_set': train_set, 'val_set': val_set}


def DataLoaderTest(config, dataset='syn', warp_input=False, export_task='train'):
    workers_test = config["data"].get('workers', 1)
    logging.info(f"Using {workers_test} workers")

    data_transforms = {
        'test': transforms.Compose([
            transforms.ToTensor(),
        ])
    }
    test_loader = None
    if dataset == 'syn':
        from datasets.synthetic_dataset import SyntheticDataset
        test_set = SyntheticDataset(
            transform=data_transforms['test'],
            train=False,
            warp_input=warp_input,
            getPts=True,
            seed=1,
            **config['data'],
        )
    elif dataset == 'hpatches':
        from datasets.patches_dataset import PatchesDataset

        test_set = PatchesDataset(
            transform=data_transforms['test'],
            **config['data'],
        )
        test_loader = torch.utils.data.DataLoader(
            test_set, batch_size=1, shuffle=False,
            pin_memory=True,
            num_workers=workers_test,
            worker_init_fn=worker_init_fn
        )
    elif dataset == 'Coco' or 'Kitti' or 'Tum':
        logging.info(f"load dataset from : {dataset}")
        Dataset = get_module(dataset, 'datasets')
        test_set = Dataset(
            export=True,
            task=export_task,
            **config['data'],
        )
        test_loader = torch.utils.data.DataLoader(
            test_set, batch_size=1, shuffle=False,
            pin_memory=True,
            num_workers=workers_test,
            worker_init_fn=worker_init_fn

        )
    else:
        raise NotImplementedError

    return {'test_set': test_set, 'test_loader': test_loader}


def camel_to_snake(name: str) -> str:
    # Add underscores between words and convert to lowercase
    snake_case = re.sub(r'(?<!^)(?=[A-Z])', '_', name).lower()
    return snake_case


def get_module(name, path: str = ""):
    path = camel_to_snake(name) if not path else f"{path}.{camel_to_snake(name)}"
    module = importlib.import_module(path)
    return getattr(module, name)


def get_model(name):
    mod = __import__('models.{}'.format(name), fromlist=[''])
    return getattr(mod, name)


def modelLoader(model='SuperPointNet', **options):
    # create model
    logging.info("=> creating model: %s", model)
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


if __name__ == '__main__':
    net = modelLoader(model='SuperPointNet')
