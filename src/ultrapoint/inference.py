import os
import torch
import argparse

from tqdm import tqdm
from pathlib import Path
from loguru import logger

from ultrapoint.models.factories import SuperPointModelsFactory
from ultrapoint.loggers.loguru import create_logger
from ultrapoint.utils.config_helpers import load_config
from ultrapoint.utils.torch_helpers import (
    clear_memory,
    set_precision,
    make_deterministic,
)
from ultrapoint.utils.utils import saveImg
from ultrapoint.utils.image_helpers import read_image
from ultrapoint.utils.draw import draw_keypoints
from ultrapoint.utils.torch_helpers import determine_device


def inference(config, images_folder: str, output_directory: str):
    clear_memory()
    make_deterministic(config["seed"])
    set_precision(config["precision"])
    device = determine_device()
    os.makedirs(output_directory, exist_ok=True)

    superpoint = SuperPointModelsFactory.create(
        model_name=config["model"]["name"],
        weights_path=config["model"]["pretrained"],
        device=device,
        **config["model"],
    )

    for filename in tqdm(Path(images_folder).iterdir(), desc="Processing images"):
        try:
            image = (
                torch.Tensor(
                    read_image(str(filename), config["data"]["preprocessing"]["resize"])
                )
                .unsqueeze(0)
                .unsqueeze(0)
                .transpose(0, 1)
                .to(device)
            )

            output = superpoint(image)
            keypoints = output["keypoints"][0].detach().cpu().numpy()
            scores = output["keypoint_scores"][0].detach().cpu().numpy()

            image = draw_keypoints(
                image.squeeze().cpu().numpy() * 255, keypoints, scores
            )
            saveImg(image, os.path.join(output_directory, f"{filename.name}.png"))

        except KeyboardInterrupt:
            clear_memory()

    logger.info(f"Processed {len(os.listdir(output_directory))} images")


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str)
    parser.add_argument("images_folder", type=str)
    parser.add_argument("output_folder", type=str)

    return parser.parse_args()


def main():
    args = parse_arguments()
    config = load_config(args.config)
    create_logger(**config["logging"])
    inference(config, args.images_folder, args.output_folder)


if __name__ == "__main__":
    main()
