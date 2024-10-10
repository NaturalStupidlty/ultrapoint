import os
import torch
import argparse

from tqdm import tqdm
from pathlib import Path
from loguru import logger

from ultrapoint.models.model_wrap import SuperPointFrontend
from ultrapoint.loggers.loguru import create_logger
from ultrapoint.utils.config_helpers import load_config
from ultrapoint.utils.torch_helpers import (
    determine_device,
    clear_memory,
    set_precision,
    make_deterministic,
)
from ultrapoint.utils.utils import inv_warp_image_batch, saveImg
from ultrapoint.utils.image_helpers import read_image
from ultrapoint.utils.draw import draw_keypoints
from ultrapoint.utils.utils import compute_valid_mask


def combine_heatmap(heatmap, inv_homographies, mask_2d, device):
    heatmap *= mask_2d
    if inv_homographies is not None:
        heatmap = inv_warp_image_batch(
            heatmap, inv_homographies[0], device=device, mode="bilinear"
        )
        mask_2d = inv_warp_image_batch(
            mask_2d, inv_homographies[0], device=device, mode="bilinear"
        )
    heatmap = torch.sum(heatmap, dim=0)
    mask_2d = torch.sum(mask_2d, dim=0)
    return heatmap / mask_2d


import numpy as np


def inference(config, images_folder: str, output_directory: str):
    clear_memory()
    make_deterministic(config["seed"])
    set_precision(config["precision"])

    device = determine_device()
    logger.info(f"Using device: {device}")
    os.makedirs(output_directory, exist_ok=True)

    nn_thresh = config["model"]["nn_thresh"]
    conf_thresh = config["model"]["detection_threshold"]
    iterations = config["data"]["homography_adaptation"]["num"]
    logger.info(f"Homography adaptation iterations: {iterations}")

    superpoint_wrapper = SuperPointFrontend(
        config=config,
        weights_path=config["pretrained"],
        nms_dist=config["model"]["nms_radius"],
        conf_thresh=conf_thresh,
        nn_thresh=nn_thresh,
        cuda=False,
        device=device,
    )
    labels_folder = images_folder.replace("all", "pseudo_labels")

    for filename in tqdm(Path(images_folder).iterdir(), desc="Processing images"):
        try:
            sample = {
                "image": torch.Tensor(
                    read_image(str(filename), config["data"]["preprocessing"]["resize"])
                ).unsqueeze(0),
                "valid_mask": compute_valid_mask(
                    torch.tensor(config["data"]["preprocessing"]["resize"]),
                    inv_homography=torch.eye(3),
                ),
            }

            heatmap = superpoint_wrapper.run(
                torch.Tensor(sample["image"]).unsqueeze(0).transpose(0, 1),
                onlyHeatmap=True,
                train=False,
            )
            outputs = combine_heatmap(
                heatmap,
                None,
                sample["valid_mask"].unsqueeze(0).transpose(0, 1).to(device),
                device,
            )
            points = superpoint_wrapper.getPtsFromHeatmap(
                outputs.detach().cpu().squeeze()
            )

            if config["model"]["subpixel"]["enable"] and points.shape[1]:
                superpoint_wrapper.heatmap = outputs
                points = superpoint_wrapper.soft_argmax_points([points])[0]

            img_pts = draw_keypoints(sample["image"].squeeze().numpy() * 255, points.T)
            corners = np.load(
                os.path.join(
                    labels_folder,
                    filename.name.replace(".jpg", "").replace(".png", "") + ".npz",
                )
            )["pts"]
            img_pts = draw_keypoints(img_pts, corners, color=(0, 0, 255))

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
