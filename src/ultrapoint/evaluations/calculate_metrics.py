import argparse
import torch

from collections import defaultdict
from tqdm import tqdm
from loguru import logger

from ultrapoint.datasets.dataloaders import DataLoadersFactory
from ultrapoint.models.factories import SuperPointModelsFactory
from ultrapoint.loggers.loguru import create_logger
from ultrapoint.utils.config_helpers import load_config
from ultrapoint.utils.metrics import compute_metrics
from ultrapoint.utils.utils import mask_to_keypoints
from ultrapoint.utils.torch_helpers import (
    clear_memory,
    set_precision,
    make_deterministic,
    determine_device,
)


def metrics_inference(config):
    clear_memory()
    make_deterministic(config["seed"])
    set_precision(config["precision"])
    device = determine_device()

    superpoint = SuperPointModelsFactory.create(
        model_name=config["model"]["name"],
        weights_path=config["model"]["pretrained"],
        device=device,
        **config["model"],
    )
    val_loader = DataLoadersFactory.create(
        config, dataset_name=config["data"]["dataset"], mode="val"
    )

    # accumulator for each metric
    metrics_acc = defaultdict(list)
    num_samples = 0

    for i, sample in enumerate(tqdm(val_loader, desc="Calculating metrics")):
        try:
            image = torch.Tensor(sample["image"]).transpose(0, 1).to(device)
            output = superpoint(image)

            keypoints = output["keypoints"][0].detach().cpu().numpy()
            scores = output["keypoint_scores"][0].detach().cpu().numpy()

            metrics = compute_metrics(
                predictions_keypoints=keypoints,
                predictions_scores=scores,
                labels_keypoints=mask_to_keypoints(sample["labels_2D"]),
            )

            metrics_acc["AP"].append(metrics["ap"])
            metrics_acc["Recall"].append(metrics["recall"])
            metrics_acc["Precision"].append(metrics["precision"])
            num_samples += 1

        except KeyboardInterrupt:
            clear_memory()
            break

    # compute means
    mean_metrics = {k: sum(v) / len(v) for k, v in metrics_acc.items()}

    logger.info(f"Processed {num_samples} images.")
    logger.info("Mean metrics over validation set:")

    for k, v in mean_metrics.items():
        logger.info(f"{k}: {v:.4f}")


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str)
    return parser.parse_args()


def main():
    args = parse_arguments()
    config = load_config(args.config)
    create_logger(**config["logging"])
    metrics_inference(config)


if __name__ == "__main__":
    main()
