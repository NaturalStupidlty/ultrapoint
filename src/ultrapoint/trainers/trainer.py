import os
import itertools
import numpy
import torch
import torch.optim
import torch.nn as nn
import torch.utils.data
import torch.optim as optim

from typing import Union
from tqdm import tqdm
from loguru import logger

from ultrapoint.utils.utils import calculate_precision_recall
from ultrapoint.loggers.tensorboard import TensorboardLogger
from ultrapoint.models.models_factory import ModelsFactory
from ultrapoint.utils.utils import labels2Dto3D
from ultrapoint.utils.torch_helpers import (
    determine_device,
    clear_memory,
    set_precision,
    make_deterministic,
)
from ultrapoint.utils.config_helpers import save_config


class Trainer:
    """
    This is the base class for training classes.
    Wrap pytorch net to help training process.
    """

    def __init__(self, config, save_path: str, device: Union[str, torch.device] = None):
        """
        default dimension:
            heatmap: torch (batch_size, H, W, 1)
            dense_desc: torch (batch_size, H, W, 256)
            pts: [batch_size, np (N, 3)]
            desc: [batch_size, np(256, N)]

        :param config:
            dense_loss, sparse_loss (default)

        :param save_path:
        :param device:
        """
        self._config = config
        self._max_iterations = config["train_iter"]
        self._batch_size = config["model"]["batch_size"]

        self._device = device if device is not None else determine_device()
        logger.info(f"Training with device: {self._device}")

        self._checkpoints_path = os.path.join(save_path, "checkpoints")
        os.makedirs(self._checkpoints_path, exist_ok=True)

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
        logger.info(f"Loaded model: {self.net.__class__.__name__}")

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

                    if self._iteration % self._config["save_interval"] == 0:
                        logger.info(f"Current iteration: {self._iteration}")
                        self.save()

                    if (
                        self._eval
                        and self._iteration % self._config["validation_interval"] == 1
                    ):
                        logger.info("Validating...")
                        for i, sample_val in enumerate(
                            itertools.islice(
                                self.val_loader, self._config.get("validation_size")
                            )
                        ):
                            self.process_sample(sample_val, self._iteration + i, "val")

                    if self._iteration > self._max_iterations:
                        logger.info("End training: {self.n_iter}")
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
                "model_state_dict": self.net.module.state_dict(),
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

        model_params = self._config["model"]["params"]
        model_name = self._config["model"]["name"]

        if self._config["pretrained"] is not None:
            checkpoint = torch.load(self._config["pretrained"])
            state_dict = checkpoint["model_state_dict"]
            self.epoch = checkpoint.get("epoch", 0)
            self._iteration = checkpoint.get("n_iter", 0)
            self._loss = checkpoint.get("loss", 0)
            optimizer_state_dict = checkpoint["optimizer_state_dict"]
        else:
            logger.info("Training model from scratch")
            state_dict = None
            optimizer_state_dict = None

        self.net = ModelsFactory.create(
            model_name=model_name, state=state_dict, **model_params
        ).to(self._device)
        self._init_optimizer(optimizer_state_dict)

        # Multi-GPU support
        logger.info(f"Using {torch.cuda.device_count()} GPUs")
        self.net = nn.DataParallel(self.net)

    def _init_optimizer(self, optimizer_state_dict=None):
        """
        Initialize or load the optimizer.
        """
        if optimizer_state_dict:
            self._optimizer.load_state_dict(optimizer_state_dict)
        else:
            self._optimizer = optim.Adam(
                self.net.parameters(),
                lr=self._config["model"]["learning_rate"],
                betas=(0.9, 0.999),
            )

    def process_sample(self, sample, iteration: int, task="val"):
        raise NotImplementedError("This method should be implemented in child classes")

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

    @staticmethod
    def precision_recall(predictions, labels):
        precision_recall_list = []
        for i in range(labels.shape[0]):
            precision_recall = calculate_precision_recall(predictions[i], labels[i])
            precision_recall_list.append(precision_recall)
        precision = numpy.mean(
            [
                precision_recall["precision"]
                for precision_recall in precision_recall_list
            ]
        )
        recall = numpy.mean(
            [precision_recall["recall"] for precision_recall in precision_recall_list]
        )
        return {"precision": precision, "recall": recall}
