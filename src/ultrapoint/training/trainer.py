import os
import numpy
import torch
import torch.optim
import torch.nn as nn
import torch.utils.data
import torch.optim as optim

from typing import Union, List
from tqdm import tqdm
from loguru import logger

from ultrapoint.loggers.tensorboard import TensorboardLogger
from ultrapoint.models.factories import SuperPointModelsFactory
from ultrapoint.utils.torch_helpers import (
    determine_device,
    clear_memory,
    set_precision,
    make_deterministic,
)
from ultrapoint.utils.config_helpers import save_config
from ultrapoint.loggers.loguru import log_scalars
from ultrapoint.utils.loss_functions.detector_loss import detector_loss
from ultrapoint.utils.metrics import compute_batch_metrics
from ultrapoint.utils.utils import labels2Dto3D, mask_to_keypoints


class Trainer:
    def __init__(self, config, save_path: str, device: Union[str, torch.device] = None):
        self._config = config
        self._max_iterations = config["train_iter"]
        self._batch_size = config["model"]["batch_size"]
        self._validation_size = self._config.get("validation_size", None)

        self._device = device if device is not None else determine_device()
        logger.info(f"Training with device: {self._device}")

        self._checkpoints_path = os.path.join(save_path, "checkpoints")
        os.makedirs(self._checkpoints_path, exist_ok=True)

        self._image_warping = self._config["data"]["warped_pair"]["enable"]
        self._det_loss_type = self._config["model"]["detector_loss"]["loss_type"]
        self._add_dustbin = True if self._det_loss_type == "softmax" else False
        self._scalar_logs = {}
        self._train = False
        self._eval = False
        self._epoch = 0
        self._loss = 0
        self._iteration = 0
        self._cell_size = 8
        self._train_loader = None
        self._val_loader = None

        # TODO: loss factory
        if self._config["model"]["dense_loss"]["enable"]:
            logger.info("Using dense loss")
            from src.ultrapoint.utils.utils import descriptor_loss

            self._desc_params = self._config["model"]["dense_loss"]["params"]
            self._descriptor_loss = descriptor_loss
            self._desc_loss_type = "dense"
        elif self._config["model"]["sparse_loss"]["enable"]:
            logger.info("Using sparse loss")
            from src.ultrapoint.utils.loss_functions.sparse_loss import (
                batch_descriptor_loss_sparse,
            )

            self._desc_params = self._config["model"]["sparse_loss"]["params"]
            self._descriptor_loss = batch_descriptor_loss_sparse
            self._desc_loss_type = "sparse"

        clear_memory()
        make_deterministic(config["seed"])
        set_precision(config["precision"])

        self._load_model()
        logger.info(f"Loaded model: {self.model.__class__.__name__}")

        self._tensorboard_logger = TensorboardLogger(save_path, config)

        self._log_important_config()
        save_config(os.path.join(save_path, "config.yaml"), config)

    def train(self):
        """
        # outer loop for training
        # control training and validation pace
        # stop when reaching max iterations
        :return:
        """
        logger.info(f"Training iterations: {self._max_iterations}")

        running_losses = []
        # TODO: pretty tqdm logs
        while self._iteration < self._max_iterations:
            try:
                logger.info(f"Epoch: {self._epoch}")
                for sample_train in tqdm(self.train_loader):
                    loss = self.process_sample(sample_train, self._iteration, "train")
                    running_losses.append(loss)

                    if self._iteration + 1 % self._config["save_interval"] == 0:
                        logger.info(f"Current iteration: {self._iteration}")
                        self.save()

                    if (
                        self._eval
                        and self._iteration + 1 % self._config["validation_interval"] == 0
                    ):
                        logger.info("Validating...")
                        for i, sample_val in enumerate(self.val_loader):
                            self.process_sample(sample_val, self._iteration + i, "val")

                    if self._iteration > self._max_iterations:
                        logger.info(f"End training: {self._max_iterations}")
                        break

                    self._iteration += 1
                self._epoch += 1

            except KeyboardInterrupt:
                logger.info("KeyboardInterrupt, saving model...")
                self.save()
        self._tensorboard_logger.close()

    def save(self):
        """
        # save checkpoint for resuming training
        :return:
        """
        filename = (
            f"{self._config['model']['name']}_{self._iteration}_checkpoint.pth.tar"
        )
        torch.save(
            {
                "epoch": self._epoch,
                "n_iter": self._iteration,
                "model_state_dict": self.model.module.state_dict(),
                "optimizer_state_dict": self._optimizer.state_dict(),
                "loss": self._loss,
            },
            os.path.join(self._checkpoints_path, filename),
        )
        logger.info(f"Saved checkpoint to {filename}")

    @property
    def train_loader(self):
        return self._train_loader

    @train_loader.setter
    def train_loader(self, loader):
        self._train = True
        self._train_loader = loader

    @property
    def val_loader(self):
        return self._val_loader

    @val_loader.setter
    def val_loader(self, loader):
        self._eval = True
        self._val_loader = loader

    def _log_important_config(self):
        logger.info(f"Learning rate: {self._config['model']['learning_rate']}")
        logger.info(f"Lambda loss: {self._config['model']['lambda_loss']}")
        logger.info(
            f"Detection threshold: {self._config['model']['detection_threshold']}",
        )
        logger.info(f"Batch size: {self._config['model']['batch_size']}")
        logger.info(f"Descriptor: {self._desc_loss_type}")
        for item in list(self._desc_params):
            logger.info(f"{item} : {self._desc_params[item]}")

    def _load_model(self):
        """
        Load model from name and params, and initialize or load optimizer.
        :return:
        """
        logger.info(f"Model: {self._config['model']['name']}")

        if self._config["pretrained"] is not None:
            checkpoint = torch.load(self._config["pretrained"], weights_only=False)
            state_dict = checkpoint["model_state_dict"]
            self.epoch = checkpoint.get("epoch", 0)
            self._iteration = checkpoint.get("n_iter", 0)
            self._loss = checkpoint.get("loss", 0)
            optimizer_state_dict = checkpoint["optimizer_state_dict"]
        else:
            logger.info("Training model from scratch")
            state_dict = None
            optimizer_state_dict = None

        self.model = SuperPointModelsFactory.create(
            model_name=self._config["model"]["name"],
            state=state_dict,
            device=self._device,
            **self._config["model"],
        )
        self._init_optimizer(optimizer_state_dict)

        # Multi-GPU support
        logger.info(f"Using {torch.cuda.device_count()} GPUs")
        self.model = nn.DataParallel(self.model)

    def _init_optimizer(self, optimizer_state_dict=None):
        """
        Initialize or load the optimizer.
        """
        if optimizer_state_dict:
            self._optimizer.load_state_dict(optimizer_state_dict)
        else:
            self._optimizer = optim.Adam(
                self.model.parameters(),
                lr=self._config["model"]["learning_rate"],
                betas=(0.9, 0.999),
            )

    def process_sample(self, sample, iteration=0, task="val"):
        assert task in ["train", "val"], "task should be either train or val"

        images, labels, mask = (
            sample["image"].to(self._device),
            sample["labels_2D"].to(self._device),
            sample["mask"],
        )
        warped_image = None
        if self._image_warping:
            warped_image, warped_labels, warped_mask = (
                sample["warped_img"].to(self._device),
                sample["warped_labels"],
                sample["warped_mask"],
            )
            mat_H, mat_H_inv = sample["homographies"], sample["inv_homographies"]

        self._optimizer.zero_grad()
        if task == "train":
            outputs = self._forward_step(images, warped_sample=warped_image)
        else:
            with torch.no_grad():
                outputs = self._forward_step(images, warped_sample=warped_image)

        labels_3D = labels2Dto3D(
            labels,
            cell_size=self._cell_size,
            add_dustbin=self._add_dustbin,
        ).float()
        mask_3D_flattened = self.get_masks(mask, self._cell_size, device=self._device)
        loss_det = detector_loss(
            predictions=outputs["detector_features"],
            labels=labels_3D.to(self._device),
            mask=mask_3D_flattened.to(self._device),
            loss_type=self._det_loss_type,
        )
        loss_det_warp = torch.tensor([0]).float().to(self._device)

        if self._image_warping:
            labels_3D = labels2Dto3D(
                warped_labels.to(self._device),
                cell_size=self._cell_size,
                add_dustbin=self._add_dustbin,
            ).float()
            mask_3D_flattened = self.get_masks(
                warped_mask, self._cell_size, device=self._device
            )
            loss_det_warp = detector_loss(
                predictions=outputs["detector_features"],
                labels=labels_3D.to(self._device),
                mask=mask_3D_flattened.to(self._device),
                loss_type=self._det_loss_type,
            )

        mask_desc = mask_3D_flattened.unsqueeze(1)
        lambda_loss = self._config["model"]["lambda_loss"]

        # descriptor loss
        if lambda_loss > 0:
            assert self._image_warping is True, "need a pair of images"
            loss_desc, mask, positive_dist, negative_dist = self._descriptor_loss(
                outputs["descriptor_features"],
                outputs["warped_descriptor_features"],
                mat_H,
                mask_valid=mask_desc,
                device=self._device,
                **self._desc_params,
            )
        else:
            ze = torch.tensor([0]).to(self._device)
            loss_desc, positive_dist, negative_dist = ze, ze, ze

        loss = loss_det + loss_det_warp
        if lambda_loss > 0:
            loss += lambda_loss * loss_desc

        self._scalar_logs.update(
            {
                "loss": loss,
                "loss_det": loss_det,
                "loss_det_warp": loss_det_warp,
                "positive_dist": positive_dist,
                "negative_dist": negative_dist,
            }
        )

        if task == "train":
            loss.backward()
            self._optimizer.step()

        if iteration % self._config["tensorboard_interval"] == 0:
            logger.info(f"Current iteration: {iteration}")
            self._log_random_sample(task, sample, outputs["keypoints"], iteration)

            preds_kpts = [kp.detach().cpu().numpy() for kp in outputs["keypoints"]]
            preds_scores = [sc.detach().cpu().numpy() for sc in outputs["keypoint_scores"]]
            gt_kpts = [mask_to_keypoints(lbl) for lbl in sample["labels_2D"]]

            # compute mean mAP / Recall@5px / Precision@5px over the batch
            mean_batch_metrics = compute_batch_metrics(
                batch_predictions_keypoints=preds_kpts,
                batch_predictions_scores=preds_scores,
                batch_labels_keypoints=gt_kpts,
                dist_thresh=5,
            )
            self._scalar_logs.update(mean_batch_metrics)
            log_scalars(self._scalar_logs, task)

        self._tensorboard_logger.log_scalars(iteration, self._scalar_logs, task)
        return loss.item()

    def _forward_step(self, sample, warped_sample: None):
        # TODO: rewrite
        outputs = self.model(sample)

        if self._image_warping:
            assert warped_sample is not None, "Forward step requires warped sample"
            warped_outputs = self.model(warped_sample)
            outputs.update(
                {
                    "warped_detector_features": warped_outputs["detector_features"],
                    "warped_descriptor_features": warped_outputs["descriptor_features"],
                }
            )

        return outputs

    def _log_random_sample(
        self,
        task: str,
        sample: dict,
        predictions_heatmap: List[torch.Tensor],
        iteration: int,
    ):
        import cv2

        random_sample_index = numpy.random.randint(0, len(sample["image"]))
        predictions = predictions_heatmap[random_sample_index].detach().cpu().numpy()
        labels = sample["labels_2D"][random_sample_index].squeeze()
        raw_image = sample["image"][random_sample_index].squeeze().numpy()
        raw_image = cv2.cvtColor(raw_image, cv2.COLOR_GRAY2RGB) * 255
        labels_image = raw_image.copy()

        labels_coordinates = numpy.where(labels > 0.0)

        self._tensorboard_logger.log_image(task, raw_image, iteration, "raw")
        for y, x in zip(*labels_coordinates):
            labels_image = cv2.circle(
                labels_image, (x, y), radius=2, color=(0, 255, 0), thickness=-1
            )
        self._tensorboard_logger.log_image(
            task,
            labels_image,
            iteration,
            "labels",
        )
        for x, y in predictions:
            labels_image = cv2.circle(
                labels_image,
                (int(x), int(y)),
                radius=1,
                color=(255, 0, 0),
                thickness=-1,
            )
        self._tensorboard_logger.log_image(
            task,
            labels_image,
            iteration,
            "predictions_labels",
        )

    @staticmethod
    def get_masks(mask_2D, cell_size, device="cpu"):
        """
        # 2D mask is constructed into 3D (Hc, Wc) space for training
        :param mask_2D:
            tensor [batch, 1, H, W]
        :param cell_size:
            8 (default)
        :param device:
        :return:
            flattened 3D mask for training
        """
        mask_3D = labels2Dto3D(
            mask_2D.to(device), cell_size=cell_size, add_dustbin=False
        ).float()
        mask_3D_flattened = torch.prod(mask_3D, 1)
        return mask_3D_flattened
