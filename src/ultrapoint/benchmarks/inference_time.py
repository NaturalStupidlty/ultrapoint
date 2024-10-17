import time
import torch
import argparse

from typing import Union
from loguru import logger

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
def benchmark_model(
    config: dict,
    n_tries: int,
    batch_size: int,
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

    times = []
    for _ in range(n_tries):
        input_tensor = torch.randn(batch_size, 1, height, width).to(device)
        start = time.time()
        model(input_tensor)
        end = time.time()
        times.append(end - start)

    return sum(times) / len(times)


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
    average_iteration_time = benchmark_model(
        config, args.n_tries, args.batch_size, device=device
    )
    logger.info(f"DEVICE: {device}")
    logger.info(f"Average inference time: {average_iteration_time:.8f} s")


if __name__ == "__main__":
    main()
