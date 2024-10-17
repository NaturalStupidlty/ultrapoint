import argparse
import os
import torch
import numpy

from tqdm import tqdm
from pathlib import Path

from ultrapoint.models.factories import SuperPointModelsFactory
from ultrapoint.utils.config_helpers import load_config
from ultrapoint.utils.draw import draw_keypoints
from ultrapoint.dataloaders import DataLoadersFactory
from ultrapoint.loggers.loguru import create_logger, logger
from ultrapoint.utils.torch_helpers import clear_memory, determine_device
from ultrapoint.utils.utils import saveImg


@torch.no_grad()
def homography_adaptation(config):
    """
    Input 1 image, output pseudo ground truth by homography adaptation with pretrained superpoint.
    Save labels:
        pred:
            'prob' (keypoints): np (N1, 3)
    """
    output_directory = os.path.join(
        Path(config["data"]["val_images_folder"]).parent, "pseudo_labels"
    )
    os.makedirs(output_directory, exist_ok=True)

    device = determine_device()
    iterations = config["data"]["homography_adaptation"]["num"]
    logger.info(f"Homography adaptation iterations: {iterations}")

    val_loader = DataLoadersFactory.create(
        config, dataset_name=config["data"]["dataset"], mode="val"
    )

    superpoint = SuperPointModelsFactory.create(
        model_name=config["model"]["name"],
        weights_path=config["model"]["pretrained"],
        **config["model"],
    ).to(device)

    for sample in tqdm(val_loader, desc="Generating pseudo labels"):
        try:
            filename = str(sample["name"][0])
            if config["skip_existing"] and os.path.exists(
                os.path.join(output_directory, f"{filename}.npz")
            ):
                logger.info(f"File {filename} exists. Skipping.")
                continue

            output = superpoint(torch.Tensor(sample["image"]).transpose(0, 1))
            keypoints = output["keypoints"][0].detach().cpu().numpy()
            scores = output["keypoint_scores"][0].detach().cpu().numpy()

            numpy.savez_compressed(
                os.path.join(output_directory, f"{filename}.npz"),
                pts=keypoints,
            )

            if not config["save_images"]:
                continue

            points = draw_keypoints(
                sample["image"].squeeze()[0].numpy() * 255, keypoints, scores
            )
            saveImg(points, os.path.join(output_directory, f"{filename}.png"))
        except KeyboardInterrupt:
            clear_memory()

    logger.info(f"Processed {len(val_loader)} samples.")


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str)

    return parser.parse_args()


def main():
    args = parse_arguments()
    config = load_config(args.config)
    create_logger(**config["logging"])
    homography_adaptation(config)


if __name__ == "__main__":
    main()
