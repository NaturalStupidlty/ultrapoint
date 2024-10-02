import argparse

from ultrapoint.dataloaders import DataLoadersFactory
from ultrapoint.trainers import TrainersFactory
from ultrapoint.utils.config_helpers import load_config
from ultrapoint.utils.logging import create_logger, logger, log_data_size
from ultrapoint.utils.utils import prepare_experiment_directory


def train(config: dict, output_directory: str):
    train_loader = DataLoadersFactory.create(
        config, config["data"]["dataset"], mode="train"
    )
    val_loader = DataLoadersFactory.create(
        config, config["data"]["dataset"], mode="test"
    )
    log_data_size(train_loader, config, tag="train")
    log_data_size(val_loader, config, tag="val")

    model_trainer = TrainersFactory.create(config, config["trainer"], output_directory)
    model_trainer.train_loader = train_loader
    model_trainer.val_loader = val_loader
    model_trainer.train()


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str)
    parser.add_argument("exper_name", help="Name of an experiment", type=str)

    return parser.parse_args()


def main():
    args = parse_arguments()
    config = load_config(args.config)
    config["logging"]["directory"] = prepare_experiment_directory(
        config["logging"]["directory"], args.exper_name
    )
    create_logger(**config["logging"])
    train(config, config["logging"]["directory"])


if __name__ == "__main__":
    main()
