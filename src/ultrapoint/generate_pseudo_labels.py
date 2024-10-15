import argparse
import os
import torch
import numpy

from tqdm import tqdm
from pathlib import Path

from ultrapoint.models.superpoint.super_point import SuperPointFrontend
from ultrapoint.models.superpoint.super_point_pretrained import SuperPoint
from ultrapoint.utils.config_helpers import load_config
from ultrapoint.utils.draw import draw_keypoints
from ultrapoint.dataloaders import DataLoadersFactory
from ultrapoint.loggers.loguru import create_logger, logger
from ultrapoint.utils.torch_helpers import (
    determine_device,
    clear_memory,
)
from ultrapoint.utils.utils import inv_warp_image_batch, saveImg


def combine_heatmap(heatmap, inv_homographies, mask_2d, device):
    heatmap *= mask_2d
    heatmap = inv_warp_image_batch(
        heatmap, inv_homographies[0], device=device, mode="bilinear"
    )
    mask_2d = inv_warp_image_batch(
        mask_2d, inv_homographies[0], device=device, mode="bilinear"
    )
    heatmap = torch.sum(heatmap, dim=0)
    mask_2d = torch.sum(mask_2d, dim=0)
    return heatmap / mask_2d


@torch.no_grad()
def homography_adaptation(config):
    """
    Input 1 image, output pseudo ground truth by homography adaptation.
    Save labels:
        pred:
            'prob' (keypoints): np (N1, 3)
    """
    device = determine_device()
    logger.info(f"Using device: {device}")

    output_directory = os.path.join(
        Path(config["data"]["val_images_folder"]).parent, "inference"
    )
    os.makedirs(output_directory, exist_ok=True)

    top_k = config["model"].get("top_k", -1)
    iterations = config["data"]["homography_adaptation"]["num"]
    logger.info(f"Homography adaptation iterations: {iterations}")

    val_loader = DataLoadersFactory.create(
        config, dataset_name=config["data"]["dataset"], mode="val"
    )
    superpoint_wrapper = SuperPointFrontend(config)

    for sample in tqdm(val_loader, desc="Generating pseudo labels"):
        try:
            filename = str(sample["name"][0])
            if config["skip_existing"] and os.path.exists(
                os.path.join(output_directory, f"{filename}.npz")
            ):
                logger.info(f"File {filename} exists. Skipping.")
                continue

            heatmap = superpoint_wrapper(sample["image"].transpose(0, 1))
            outputs = combine_heatmap(
                heatmap,
                sample["homographies"].to(device),
                sample["mask"].transpose(0, 1).to(device),
                device,
            )
            points = superpoint_wrapper.heatmap_to_keypoints(
                outputs.detach().cpu().squeeze()
            )

            numpy.savez_compressed(
                os.path.join(output_directory, f"{filename}.npz"),
                pts=points.transpose()[:top_k, :],
            )

            if not config["save_images"]:
                continue

            img_pts = draw_keypoints(sample["image_2D"].numpy().squeeze() * 255, points)
            saveImg(img_pts, os.path.join(output_directory, f"{filename}.png"))
        except KeyboardInterrupt:
            clear_memory()

    logger.info(f"Processed {len(val_loader)} samples.")


@torch.no_grad()
def homography_adaptation_pretrained(config):
    """
    Input 1 image, output pseudo ground truth by homography adaptation with pretrained superpoint.
    Save labels:
        pred:
            'prob' (keypoints): np (N1, 3)
    """
    device = determine_device()
    logger.info(f"Using device: {device}")

    output_directory = os.path.join(
        Path(config["data"]["val_images_folder"]).parent, "pseudo_labels"
    )
    os.makedirs(output_directory, exist_ok=True)

    top_k = config["model"].get("top_k", -1)
    iterations = config["data"]["homography_adaptation"]["num"]
    logger.info(f"Homography adaptation iterations: {iterations}")

    val_loader = DataLoadersFactory.create(
        config, dataset_name=config["data"]["dataset"], mode="val"
    )

    superpoint = SuperPoint(**config["model"]).to(device)
    superpoint.load_state_dict(torch.load(config["pretrained"], weights_only=False))

    for sample in tqdm(val_loader, desc="Generating pseudo labels"):
        try:
            filename = str(sample["name"][0])
            if config["skip_existing"] and os.path.exists(
                os.path.join(output_directory, f"{filename}.npz")
            ):
                logger.info(f"File {filename} exists. Skipping.")
                continue

            sample["image"] = sample["image"].transpose(0, 1).to(device)
            outputs = superpoint(sample)
            points = (
                torch.cat(
                    (
                        outputs["keypoints"][0],
                        outputs["keypoint_scores"][0].unsqueeze(1),
                    ),
                    dim=1,
                )
                .cpu()
                .numpy()
            )

            points = points[points[:, 2] > 0.2]

            numpy.savez_compressed(
                os.path.join(output_directory, f"{filename}.npz"),
                pts=points[:top_k, :],
            )

            if not config["save_images"]:
                continue

            img_pts = draw_keypoints(
                sample["image_2D"].numpy().squeeze() * 255,
                points.transpose(),
            )
            saveImg(img_pts, os.path.join(output_directory, f"{filename}.png"))
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
    if config["model"]["name"] == "SuperPointPretrained":
        homography_adaptation_pretrained(config)
    else:
        homography_adaptation(config)


if __name__ == "__main__":
    main()
