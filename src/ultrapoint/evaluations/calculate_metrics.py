import argparse

from collections import defaultdict
from tqdm import tqdm
from loguru import logger

from ultrapoint.datasets.dataloaders import DataLoadersFactory
from ultrapoint.models.factories import SuperPointModelsFactory
from ultrapoint.loggers.loguru import create_logger
from ultrapoint.utils.config_helpers import load_config
from ultrapoint.utils.metrics import compute_batch_metrics
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

    metrics_acc = defaultdict(list)
    total_images = 0

    for batch in tqdm(val_loader, desc="Calculating metrics"):
        try:
            output = superpoint(batch["image"].to(device))

            B = batch["image"].size(0)
            # collect per-sample preds & gts
            preds_kpts = [ output["keypoints"][i].detach().cpu().numpy()
                            for i in range(B) ]
            preds_scores = [ output["keypoint_scores"][i].detach().cpu().numpy()
                             for i in range(B) ]
            gt_kpts     = [ mask_to_keypoints(batch["labels_2D"][i])
                            for i in range(B) ]

            # compute this batch's mean AP / R@0.5 / P@0.5
            batch_metrics = compute_batch_metrics(
                batch_predictions_keypoints=preds_kpts,
                batch_predictions_scores=preds_scores,
                batch_labels_keypoints=gt_kpts,
                dist_thresh=config.get("eval_dist_thresh", 5),
            )

            metrics_acc["mAP"].append(batch_metrics["mAP"])
            metrics_acc["Recall"].append(batch_metrics["Recall"])
            metrics_acc["Precision"].append(batch_metrics["Precision"])
            total_images += B

        except KeyboardInterrupt:
            clear_memory()
            break

    # average over all batches (assuming roughly equal batch sizes)
    mean_metrics = { k: sum(v) / len(v) for k, v in metrics_acc.items() }

    logger.info(f"Processed {total_images} images in {len(metrics_acc['AP'])} batches.")
    logger.info("Mean metrics over validation set:")
    logger.info(f"  AP:        {mean_metrics['AP']:.4f}")
    logger.info(f"  Recall@0.5:{mean_metrics['Recall']:.4f}")
    logger.info(f"  Precision@0.5: {mean_metrics['Precision']:.4f}")


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
