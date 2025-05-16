import argparse

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
    superpoint.eval()  # Ensure model is in eval mode

    val_loader = DataLoadersFactory.create(
        config, dataset_name=config["data"]["dataset"], mode="test"
    )

    total_images = 0
    total_mAP = 0.0
    total_recall = 0.0
    total_precision = 0.0

    for batch in tqdm(val_loader, desc="Calculating metrics"):
        try:
            B = batch["image"].size(0)
            total_images += B

            output = superpoint(batch["image"].to(device))

            # collect per-sample preds & gts
            preds_kpts = [ output["keypoints"][i].detach().cpu().numpy()
                            for i in range(B) ]
            preds_scores = [ output["keypoint_scores"][i].detach().cpu().numpy()
                             for i in range(B) ]
            gt_kpts     = [ mask_to_keypoints(batch["labels_2D"][i])
                            for i in range(B) ]

            # compute this batch's metrics
            batch_metrics = compute_batch_metrics(
                batch_predictions_keypoints=preds_kpts,
                batch_predictions_scores=preds_scores,
                batch_labels_keypoints=gt_kpts,
                dist_thresh=config.get("eval_dist_thresh", 5),
            )

            # weight metrics by number of samples in batch
            total_mAP += batch_metrics["mAP"] * B
            total_recall += batch_metrics["Recall"] * B
            total_precision += batch_metrics["Precision"] * B

            # Optionally log batch size and batch-level metrics
            # logger.debug(f"Batch size: {B}, mAP: {batch_metrics['mAP']:.4f}")

        except KeyboardInterrupt:
            clear_memory()
            break

    if total_images == 0:
        logger.warning("No images processed.")
        return

    mean_metrics = {
        "mAP": total_mAP / total_images,
        "Recall": total_recall / total_images,
        "Precision": total_precision / total_images,
    }

    logger.info(f"Processed {total_images} images.")
    logger.info("Mean metrics over validation set:")
    logger.info(f"  AP:            {mean_metrics['mAP']:.4f}")
    logger.info(f"  Recall@0.5:    {mean_metrics['Recall']:.4f}")
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
