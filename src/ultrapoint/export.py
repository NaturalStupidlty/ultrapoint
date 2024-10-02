import argparse
import dotenv
import os

import numpy as np

from tqdm import tqdm
from tensorboardX import SummaryWriter

from src.ultrapoint.utils.utils import prepare_experiment_directory
from src.ultrapoint.utils.logging import create_logger, logger, log_data_size
from src.ultrapoint.utils.loader import DataLoaderTest
from src.ultrapoint.utils.loader import get_module
from src.ultrapoint.utils.config_helpers import load_config, save_config
from src.ultrapoint.utils.torch_helpers import (
    make_deterministic,
    set_precision,
    determine_device,
)
from src.ultrapoint.utils.loader import get_checkpoints_path
from src.ultrapoint.utils.torch_helpers import squeeze_to_numpy
from src.ultrapoint.models.model_wrap import PointTracker


def get_keypoints(model, img, device, subpixel: bool = None, patch_size: int = None):
    """
    pts: list [numpy (3, N)]
    desc: list [numpy (256, N)]
    """
    assert subpixel is None or patch_size is not None

    # heatmap: numpy [batch, 1, H, W]
    heatmap = model.run(img.to(device))
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
    device = determine_device()
    logger.info(f"Training with device: {device}")
    save_config(os.path.join(output_directory, "config.yaml"), config)

    predictions_folder = os.path.join(
        get_checkpoints_path(output_directory), "predictions"
    )
    os.makedirs(predictions_folder, exist_ok=True)

    # data loading
    data = DataLoaderTest(config, dataset=config["data"]["dataset"])
    test_set, test_loader = data["test_set"], data["test_loader"]
    log_data_size(test_loader, config, tag="test")

    # model loading
    model_heatmap = get_module(config["front_end_model"])
    model_agent = model_heatmap(config["model"], device=device)
    model_agent.loadModel()
    model_agent.writer = SummaryWriter(output_directory)

    # tracker
    tracker = PointTracker(max_length=2, nn_thresh=model_agent.nn_thresh)

    subpixel = config["model"]["subpixel"]["enable"]
    patch_size = config["model"]["subpixel"]["patch_size"]

    for sample_index, sample in tqdm(enumerate(test_loader)):
        image_0, image_1 = sample["image"], sample["warped_image"]
        points0, descriptors0 = get_keypoints(
            model_agent, image_0, device, subpixel, patch_size
        )
        tracker.update(points0, descriptors0)
        points1, descriptors1 = get_keypoints(
            model_agent, image_1, device, subpixel, patch_size
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

    logger.info(f"output pairs: {len(test_loader)}")


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str)
    parser.add_argument("exper_name", type=str)
    parser.add_argument("--eval", action="store_true")
    parser.add_argument(
        "--debug", action="store_true", default=False, help="turn on debuging mode"
    )
    parser.set_defaults(func=export_descriptors)

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
