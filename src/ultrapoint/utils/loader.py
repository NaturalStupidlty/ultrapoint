"""many loaders
# loader for model, dataset, testing dataset
"""
import os
from loguru import logger


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
