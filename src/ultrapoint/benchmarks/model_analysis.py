import torch
import argparse

from typing import Union
from loguru import logger
from fvcore.nn import FlopCountAnalysis, parameter_count_table

from ultrapoint.models.factories import SuperPointModelsFactory
from ultrapoint.loggers.loguru import create_logger
from ultrapoint.utils.config_helpers import load_config
from ultrapoint.utils.torch_helpers import (
    clear_memory,
    set_precision,
    make_deterministic,
    determine_device,
)


@torch.no_grad()
def get_model_stats(
    config: dict,
    device: Union[str, torch.device] = "cuda",
):
    model = SuperPointModelsFactory.create(
        model_name=config["model"]["name"],
        weights_path=config["model"]["pretrained"],
        device=device,
        **config["model"],
    )
    model.to(device)
    model.eval()

    height, width = config["data"]["preprocessing"]["resize"]
    input_tensor = torch.randn(1, 1, height, width).to(device)

    flops = FlopCountAnalysis(model, input_tensor)
    logger.info(f"FLOPs: {flops.total()}")
    logger.info(parameter_count_table(model))


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str)
    parser.add_argument("--n_tries", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=1)
    return parser.parse_args()


def main():
    args = parse_arguments()
    config = load_config(args.config)
    create_logger(**config["logging"])
    clear_memory()
    make_deterministic(config["seed"])
    set_precision(config["precision"])
    device = determine_device()
    get_model_stats(config, device)


if __name__ == "__main__":
    main()
