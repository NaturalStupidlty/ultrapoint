import os
import torch
import argparse

from tqdm import tqdm
from pathlib import Path
from loguru import logger

from ultrapoint.models.models_factory import ModelsFactory
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
from ultrapoint.utils.utils import compute_mask
from ultrapoint.utils.torch_helpers import determine_device


def inference(config, images_folder: str, output_directory: str):
    clear_memory()
    make_deterministic(config["seed"])
    set_precision(config["precision"])
    device = determine_device()

    os.makedirs(output_directory, exist_ok=True)
    iterations = config["data"]["homography_adaptation"]["num"]
    logger.info(f"Homography adaptation iterations: {iterations}")

    state_dict = torch.load(config["pretrained"], map_location=device)[
        "model_state_dict"
    ]
    superpoint = ModelsFactory.create(
        model_name="SuperPoint", config=config, state=state_dict
    ).to(device)

    for filename in tqdm(Path(images_folder).iterdir(), desc="Processing images"):
        try:
            sample = {
                "image": torch.Tensor(
                    read_image(str(filename), config["data"]["preprocessing"]["resize"])
                ).unsqueeze(0),
                "mask": compute_mask(
                    torch.tensor(config["data"]["preprocessing"]["resize"]),
                    inv_homography=torch.eye(3),
                ),
            }

            output = superpoint(
                torch.Tensor(sample["image"]).unsqueeze(0).transpose(0, 1)
            )
            keypoints = output["keypoints"][0].detach().cpu().numpy()
            scores = output["keypoint_scores"][0].detach().cpu().numpy()

            img_pts = draw_keypoints(
                sample["image"].squeeze().numpy() * 255, keypoints, scores
            )
            saveImg(img_pts, os.path.join(output_directory, f"{filename.name}.png"))

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
