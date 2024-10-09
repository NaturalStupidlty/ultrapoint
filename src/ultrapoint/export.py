import argparse
import os

import numpy as np

from tqdm import tqdm

from ultrapoint.trainers import TrainersFactory
from ultrapoint.utils.utils import prepare_experiment_directory
from ultrapoint.loggers.loguru import create_logger, logger, log_data_size
from ultrapoint.dataloaders import DataLoadersFactory
from ultrapoint.utils.config_helpers import load_config
from ultrapoint.utils.torch_helpers import squeeze_to_numpy
from ultrapoint.models.model_wrap import PointTracker


def get_keypoints(model, img, subpixel: bool = None, patch_size: int = None):
    """
    pts: list [numpy (3, N)]
    desc: list [numpy (256, N)]
    """
    assert subpixel is None or patch_size is not None

    # heatmap: numpy [batch, 1, H, W]
    heatmap = model.run(img.to(model._device))
    points = model.heatmap_to_pts(heatmap)

    if subpixel:
        points = model.soft_argmax_points(points, patch_size=patch_size)

    desc_sparse = model.sparsify_descriptors()
    print("points[0]: ", points[0].shape, ", desc_sparse[0]: ", desc_sparse[0].shape)
    print("points[0]: ", points[0].shape)

    return points[0], desc_sparse[0]


def export_descriptors(config, output_directory):
    """
    # input 2 images, output keypoints and correspondence
    save prediction:
        pred:
            'image': np(320,240)
            'prob' (keypoints): np (N1, 2)
            'desc': np (N2, 256)
            'warped_image': np(320,240)
            'warped_prob' (keypoints): np (N2, 2)
            'warped_desc': np (N2, 256)
            'homography': np (3,3)
            'matches': np [N3, 4]
    """
    predictions_folder = os.path.join(output_directory, "predictions")
    os.makedirs(predictions_folder, exist_ok=True)

    val_loader = DataLoadersFactory.create(
        config, dataset_name=config["data"]["dataset"], mode="val"
    )
    log_data_size(val_loader, config, tag="val")

    model_trainer = TrainersFactory.create(config, config["trainer"], output_directory)
    model_trainer.val_loader = val_loader

    # tracker
    tracker = PointTracker(max_length=2, nn_thresh=model_trainer.nn_thresh)

    subpixel = config["model"]["subpixel"]["enable"]
    patch_size = config["model"]["subpixel"]["patch_size"]

    for sample_index, sample in tqdm(enumerate(val_loader)):
        image_0, image_1 = sample["image"], sample["warped_image"]
        points0, descriptors0 = get_keypoints(
            model_trainer, image_0, subpixel, patch_size
        )
        tracker.update(points0, descriptors0)
        points1, descriptors1 = get_keypoints(
            model_trainer, image_1, subpixel, patch_size
        )
        tracker.update(points1, descriptors1)
        matches = tracker.get_matches()
        tracker.clear_desc()

        predictions = {
            "image": squeeze_to_numpy(image_0),
            "prob": points0.transpose(),
            "desc": descriptors0.transpose(),
            "warped_image": squeeze_to_numpy(image_1),
            "warped_prob": points1.transpose(),
            "warped_desc": descriptors1.transpose(),
            "homography": squeeze_to_numpy(sample["homography"]),
            "matches": matches.transpose(),
        }

        path = os.path.join(predictions_folder, f"{sample_index}.npz")
        np.savez_compressed(path, **predictions)
        logger.debug(f"Saved predictions to: {path}")

    logger.info(f"output pairs: {len(val_loader)}")


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str)
    parser.add_argument("exper_name", type=str)

    return parser.parse_args()


def main():
    args = parse_arguments()
    config = load_config(args.config)
    config["logging"]["directory"] = prepare_experiment_directory(
        config["logging"]["directory"], args.exper_name
    )
    create_logger(**config["logging"])
    export_descriptors(config, config["logging"]["directory"])


if __name__ == "__main__":
    main()
