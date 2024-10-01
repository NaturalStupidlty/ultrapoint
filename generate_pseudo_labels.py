import argparse
import dotenv
import os

import numpy as np
import torch
import torch.optim
import torch.utils.data

from tqdm import tqdm
from pathlib import Path

from utils.utils import inv_warp_image_batch
from utils.logging import create_logger, logger
from utils.loader import DataLoaderTest
from utils.utils import saveImg
from utils.draw import draw_keypoints
from utils.config_helpers import load_config, save_config
from utils.torch_helpers import make_deterministic, set_precision, determine_device
from models.model_wrap import SuperPointFrontend_torch


def combine_heatmap(heatmap, inv_homographies, mask_2d, device):
    heatmap = heatmap * mask_2d
    heatmap = inv_warp_image_batch(
        heatmap, inv_homographies[0, :, :, :], device=device, mode="bilinear"
    )

    mask_2d = inv_warp_image_batch(
        mask_2d, inv_homographies[0, :, :, :], device=device, mode="bilinear"
    )
    heatmap = torch.sum(heatmap, dim=0)
    mask_2d = torch.sum(mask_2d, dim=0)

    return heatmap / mask_2d


@torch.no_grad()
def export_detector_homo_adapt_gpu(config, output_images: bool = True):
    """
    input 1 images, output pseudo ground truth by homography adaptation.
    Save labels:
        pred:
            'prob' (keypoints): np (N1, 3)
    """
    device = determine_device()
    logger.info(f"Training with device: {device}")
    output_directory = os.path.join(config["data"]["data_path"], "labels")
    os.makedirs(output_directory, exist_ok=True)
    save_config(os.path.join(output_directory, "config.yaml"), config)

    # parameters
    top_k = config["model"]["top_k"]
    nn_thresh = config["model"]["nn_thresh"]
    conf_thresh = config["model"]["detection_threshold"]
    iterations = config["data"]["homography_adaptation"]["num"]

    logger.info(f"=> will save everything to {output_directory}")
    os.makedirs(output_directory, exist_ok=True)

    # data loading
    data = DataLoaderTest(config, dataset=config["data"]["dataset"])
    test_set, test_loader = data["test_set"], data["test_loader"]

    # load pretrained model
    path = config["pretrained"]
    try:
        logger.info(f"==> Loading pre-trained network {path}")
        # This class runs the SuperPoint network and processes its outputs.
        fe = SuperPointFrontend_torch(
            config=config,
            weights_path=path,
            nms_dist=config["model"]["nms"],
            conf_thresh=conf_thresh,
            nn_thresh=nn_thresh,
            cuda=False,
            device=device,
        )
        logger.info("==> Successfully loaded pre-trained network.")

        fe.net_parallel()
        fe.net.eval()
    except Exception:
        logger.error(f"Loading model: {path} failed!")
        raise

    logger.info(f"Homography adaptation: {iterations}\n")

    for sample in tqdm(test_loader):
        img, mask_2d = sample["image"], sample["valid_mask"]
        img = img.transpose(0, 1)
        img_2d = sample["image_2D"].numpy().squeeze()
        mask_2d = mask_2d.transpose(0, 1)

        inv_homographies, homographies = (
            sample["homographies"],
            sample["inv_homographies"],
        )
        img, mask_2d, homographies, inv_homographies = (
            img.to(device),
            mask_2d.to(device),
            homographies.to(device),
            inv_homographies.to(device),
        )

        filename = str(sample["name"][0])
        p = os.path.join(output_directory, f"{sample['name'][0]}.npz")
        if os.path.exists(p):
            logger.info(f"File {filename} exists. Skipping.")
            continue

        # pass through network
        heatmap = fe.run(img, onlyHeatmap=True, train=False)
        outputs = combine_heatmap(heatmap, inv_homographies, mask_2d, device)
        pts = fe.getPtsFromHeatmap(outputs.detach().cpu().squeeze())  # (x,y, prob)

        # subpixel prediction
        if config["model"]["subpixel"]["enable"]:
            fe.heatmap = outputs  # tensor [batch, 1, H, W]
            print("outputs: ", outputs.shape)
            print("pts: ", pts.shape)
            pts = fe.soft_argmax_points([pts])
            pts = pts[0]

        # top K points
        pts = pts.transpose()
        print("total points: ", pts.shape)
        print("pts: ", pts[:5])
        if top_k:
            if pts.shape[0] > top_k:
                pts = pts[:top_k, :]
                print("topK filter: ", pts.shape)

        # save keypoints
        pred = {}
        pred.update({"pts": pts})

        # make directories
        if config["data"]["dataset"] == "Kitti" or "Kitti_inh":
            scene_name = sample["scene_name"][0]
            os.makedirs(Path(output_directory, scene_name), exist_ok=True)

        np.savez_compressed(os.path.join(output_directory, f"{filename}.npz"), **pred)

        # output images for visualization labels
        if output_images:
            img_pts = draw_keypoints(img_2d * 255, pts.transpose())
            saveImg(img_pts, os.path.join(output_directory, f"{filename}.png"))

    logger.info(f"Output pairs: {len(test_loader)}\n")


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str)
    parser.add_argument("--eval", action="store_true")
    parser.add_argument(
        "--debug", action="store_true", default=False, help="turn on debuging mode"
    )
    parser.set_defaults(func=export_detector_homo_adapt_gpu)

    return parser.parse_args()


def main():
    dotenv.load_dotenv()
    args = parse_arguments()
    config = load_config(args.config)
    set_precision(config["precision"])
    make_deterministic(config["seed"])
    create_logger(**config["logging"])
    args.func(config)


if __name__ == "__main__":
    main()
