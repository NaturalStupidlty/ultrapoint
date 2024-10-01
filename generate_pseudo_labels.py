import argparse
import dotenv
import os
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from utils.utils import inv_warp_image_batch, saveImg
from utils.logging import create_logger, logger
from utils.loader import DataLoaderTest
from utils.draw import draw_keypoints
from utils.config_helpers import load_config, save_config
from utils.torch_helpers import make_deterministic, set_precision, determine_device
from models.model_wrap import SuperPointFrontend_torch


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
def homography_adaptation(config, output_images: bool = True):
    """
    Input 1 image, output pseudo ground truth by homography adaptation.
    Save labels:
        pred:
            'prob' (keypoints): np (N1, 3)
    """
    device = determine_device()
    logger.info(f"Training with device: {device}")

    output_directory = os.path.join(config["data"]["data_path"], "pseudo_labels")
    os.makedirs(output_directory, exist_ok=True)
    save_config(os.path.join(output_directory, "config.yaml"), config)

    top_k = config["model"]["top_k"]
    nn_thresh = config["model"]["nn_thresh"]
    conf_thresh = config["model"]["detection_threshold"]
    iterations = config["data"]["homography_adaptation"]["num"]

    data = DataLoaderTest(config, dataset=config["data"]["dataset"])
    test_loader = data["test_loader"]

    path = config["pretrained"]
    try:
        logger.info(f"Loading pre-trained network {path}")
        superpoint_wrapper = SuperPointFrontend_torch(
            config=config,
            weights_path=path,
            nms_dist=config["model"]["nms"],
            conf_thresh=conf_thresh,
            nn_thresh=nn_thresh,
            cuda=False,
            device=device,
        )
        logger.info("Successfully loaded pre-trained network.")
    except Exception as e:
        logger.error(f"Failed to load model: {path} - {e}")
        raise

    logger.info(f"Homography adaptation iterations: {iterations}")

    for sample in tqdm(test_loader, desc="Generating pseudo labels"):
        img = sample["image"].transpose(0, 1).to(device)
        mask_2d = sample["valid_mask"].transpose(0, 1).to(device)
        homographies = sample["homographies"].to(device)

        filename = str(sample["name"][0])
        if config["skip_existing"] and os.path.exists(
            os.path.join(output_directory, f"{filename}.npz")
        ):
            logger.info(f"File {filename} exists. Skipping.")
            continue

        heatmap = superpoint_wrapper.run(img, onlyHeatmap=True, train=False)
        outputs = combine_heatmap(heatmap, homographies, mask_2d, device)
        points = superpoint_wrapper.getPtsFromHeatmap(outputs.detach().cpu().squeeze())

        if config["model"]["subpixel"]["enable"]:
            superpoint_wrapper.heatmap = outputs
            points = superpoint_wrapper.soft_argmax_points([points])[0]

        if top_k and points.shape[0] > top_k:
            points = points[:top_k]

        pred = {"pts": points}
        if config["data"]["dataset"] in ["Kitti", "Kitti_inh"]:
            scene_name = sample["scene_name"][0]
            os.makedirs(Path(output_directory, scene_name), exist_ok=True)

        np.savez_compressed(os.path.join(output_directory, f"{filename}.npz"), **pred)

        if output_images:
            img_pts = draw_keypoints(
                sample["image_2D"].numpy().squeeze() * 255, points.transpose()
            )
            saveImg(img_pts, os.path.join(output_directory, f"{filename}.png"))

    logger.info(f"Processed {len(test_loader)} samples.")


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str)
    parser.add_argument("--eval", action="store_true")
    parser.add_argument(
        "--debug", action="store_true", default=False, help="Enable debugging mode"
    )
    return parser.parse_args()


def main():
    dotenv.load_dotenv()
    args = parse_arguments()
    config = load_config(args.config)
    set_precision(config["precision"])
    make_deterministic(config["seed"])
    create_logger(**config["logging"])
    homography_adaptation(config)


if __name__ == "__main__":
    main()
