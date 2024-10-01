import argparse
import dotenv
import os

from tensorboardX import SummaryWriter
from ultrapoint.utils.utils import prepare_experiment_directory
from ultrapoint.utils.loader import DataLoader, get_checkpoints_path, get_module
from ultrapoint.utils.logging import create_logger, logger, log_data_size
from ultrapoint.utils.torch_helpers import make_deterministic, set_precision, determine_device
from ultrapoint.utils.config_helpers import load_config, save_config


def train(config: dict, output_directory: str):
    device = determine_device()
    logger.info(f"Training with device: {device}")

    save_config(os.path.join(output_directory, "config.yaml"), config)

    data = DataLoader(config, dataset=config["data"]["dataset"])
    train_loader, val_loader = data["train_loader"], data["val_loader"]
    log_data_size(train_loader, config, tag="train")
    log_data_size(val_loader, config, tag="val")

    # init the training agent using config file
    train_model_frontend = get_module(config["front_end_model"])
    train_agent = train_model_frontend(
        config, save_path=get_checkpoints_path(output_directory), device=device
    )

    # writer from tensorboard
    train_agent.writer = SummaryWriter(output_directory)

    # feed the data into the agent
    train_agent.train_loader = train_loader
    train_agent.val_loader = val_loader

    # load model initiates the model and load the pretrained model (if any)
    train_agent.loadModel()
    train_agent.dataParallel()

    try:
        train_agent.train()
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt, saving model...")
        train_agent.saveModel()
        pass


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
    set_precision(config["precision"])
    make_deterministic(config["seed"])
    output_directory = prepare_experiment_directory(
        os.getenv("EXPER_PATH"), args.exper_name
    )
    create_logger(**config["logging"], logs_dir=output_directory)
    args.func(config, output_directory)


if __name__ == "__main__":
    main()
