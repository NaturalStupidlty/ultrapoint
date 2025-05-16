import argparse
import os
import torch
import numpy as np

from tqdm import tqdm

from ultrapoint.models.factories import SuperPointModelsFactory
from ultrapoint.loggers.loguru import create_logger, logger
from ultrapoint.datasets.dataloaders import DataLoadersFactory
from ultrapoint.utils.config_helpers import load_config
from ultrapoint.evaluations.point_tracker import PointTracker
from ultrapoint.utils.torch_helpers import determine_device, squeeze_to_numpy


@torch.no_grad()
def export_descriptors(config, output_directory):
    device = determine_device()
    os.makedirs(output_directory, exist_ok=True)

    val_loader = DataLoadersFactory.create(
        config, dataset_name=config["data"]["dataset"], mode="val"
    )
    superpoint = SuperPointModelsFactory.create(
        model_name=config["model"]["name"],
        weights_path=config["model"]["pretrained"],
        device=device,
        **config["model"],
    )
    tracker = PointTracker(**config["tracker"])

    def extract_features(image: np.ndarray, device: torch.device):
        image = torch.Tensor(image).transpose(0, 1).to(device)
        output = superpoint(image)
        return (
            output["keypoints"][0].detach().cpu().numpy(),
            output["descriptors"][0].detach().cpu().numpy(),
        )

    for sample_index, sample in tqdm(enumerate(val_loader)):
        points0, descriptors0 = extract_features(sample["image"], device)
        tracker.update(points0, descriptors0)

        points1, descriptors1 = extract_features(sample["warped_image"], device)
        tracker.update(points1, descriptors1)
        matches = tracker.get_matches()
        tracker.clear_desc()

        predictions = {
            "image": squeeze_to_numpy(sample["image"]),
            "keypoints": points0,
            "descriptor": descriptors0,
            "warped_image": sample["warped_image"],
            "warped_keypoints": points1,
            "warped_descriptor": descriptors1,
            "homography": squeeze_to_numpy(sample["homography"]),
            "matches": matches,
        }

        path = os.path.join(output_directory, f"{sample_index}.npz")
        np.savez_compressed(path, **predictions)
        logger.debug(f"Saved predictions to: {path}")
        torch.cuda.empty_cache()

    logger.info(f"Descriptors exported to: {output_directory}")


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str)
    parser.add_argument(
        "--output_directory", type=str, default="../assets/exported_descriptors"
    )

    return parser.parse_args()


def main():
    args = parse_arguments()
    config = load_config(args.config)
    create_logger(**config["logging"])
    export_descriptors(config, args.output_directory)


if __name__ == "__main__":
    main()
