import torch
import argparse

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
    model: torch.nn.Module,
    input_shape: tuple,
    n_tries: int = 1000,
    n_warmup: int = 20,
    device: torch.device = torch.device("cuda"),
):
    # Move model to device and set eval mode
    model.to(device)
    model.eval()

    # Pre-generate randomized inputs on GPU
    inputs = [torch.randn(input_shape, device=device) for _ in range(n_tries + n_warmup)]

    # Warm-up iterations (excluded from timing)
    for i in range(n_warmup):
        _ = model(inputs[i])
    torch.cuda.synchronize()

    # Use CUDA events for precise timing
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    times_ms = []

    for i in range(n_warmup, n_warmup + n_tries):
        start_event.record()
        model(inputs[i])
        end_event.record()
        torch.cuda.synchronize()
        times_ms.append(start_event.elapsed_time(end_event))

    avg_time_ms = sum(times_ms) / len(times_ms)
    print(f"Average inference time per batch: {avg_time_ms:.6f} ms")
    return avg_time_ms / 1000  # Return in seconds


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
    model = SuperPointModelsFactory.create(
        model_name=config["model"]["name"],
        weights_path=config["model"]["pretrained"],
        device="cuda",
        **config["model"],
    )

    # Resize from config
    H, W = config["data"]["preprocessing"]["resize"]
    batch_size = 1

    # Benchmark
    average_iteration_time = benchmark_model(
        model,
        input_shape=(batch_size, 1, H, W),
        n_tries=args.n_tries,
        n_warmup=100,
        device=torch.device("cuda"),
    )
    logger.info(f"DEVICE: {device}")
    logger.info(f"Average inference time: {average_iteration_time:.8f} s")


if __name__ == "__main__":
    main()
