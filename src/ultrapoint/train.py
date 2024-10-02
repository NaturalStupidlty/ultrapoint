import argparse
import os

import dotenv
from tensorboardX import SummaryWriter

from ultrapoint.utils.config_helpers import load_config, save_config
from ultrapoint.utils.loader import DataLoadersFabric, get_checkpoints_path, get_module
from ultrapoint.utils.logging import create_logger, logger, log_data_size
from ultrapoint.utils.torch_helpers import (
    make_deterministic,
    set_precision,
    determine_device,
    clear_memory,
)
from ultrapoint.utils.utils import prepare_experiment_directory


def train(config: dict, output_directory: str):
    device = determine_device()
    logger.info(f"Training with device: {device}")

    save_config(os.path.join(output_directory, "config.yaml"), config)

    train_loader = DataLoadersFabric.create(
        config, dataset=config["data"]["dataset"], mode="train"
    )
    val_loader = DataLoadersFabric.create(
        config, dataset=config["data"]["dataset"], mode="test"
    )
    log_data_size(train_loader, config, tag="train")
    log_data_size(val_loader, config, tag="val")

    train_model_frontend = get_module(config["front_end_model"])
    train_agent = train_model_frontend(
        config, save_path=get_checkpoints_path(output_directory), device=device
    )

    train_agent.writer = SummaryWriter(output_directory)
    train_agent.train_loader = train_loader
    train_agent.val_loader = val_loader
    train_agent.loadModel()
    train_agent.dataParallel()

    try:
        train_agent.train()
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt, saving model...")
        train_agent.saveModel()
    finally:
        train_agent.writer.close()
        clear_memory()


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str)
    parser.add_argument("exper_name", help="Name of an experiment", type=str)
    parser.add_argument("--eval", action="store_true")
    parser.add_argument(
        "--debug", action="store_true", default=False, help="turn on debuging mode"
    )
    parser.set_defaults(func=train)

    return parser.parse_args()


def main():
    dotenv.load_dotenv()
    args = parse_arguments()
    config = load_config(args.config)
    clear_memory()
    set_precision(config["precision"])
    make_deterministic(config["seed"])
    output_directory = prepare_experiment_directory(
        os.getenv("EXPER_PATH"), args.exper_name
    )
    create_logger(**config["logging"], logs_dir=output_directory)
    args.func(config, output_directory)


if __name__ == "__main__":
    main()
