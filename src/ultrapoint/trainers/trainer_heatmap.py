import numpy
import numpy as np
import torch
import torch.optim
import torch.utils.data

from loguru import logger
from ultrapoint.utils.utils import (
    flattenDetection,
    labels2Dto3D,
)
from ultrapoint.trainers.trainer import Trainer
from ultrapoint.loggers.loguru import log_losses
from ultrapoint.utils.loss_functions.detector_loss import detector_loss
from ultrapoint.utils.torch_helpers import torch_to_numpy
from ultrapoint.utils.losses import (
    do_log,
    norm_patches,
    extract_patches,
    soft_argmax_2d,
)


class TrainerHeatmap(Trainer):
    """
    heatmap: torch (batch_size, H, W, 1)
    dense_desc: torch (batch_size, H, W, 256)
    pts: [batch_size, np (N, 3)]
    descriptor: [batch_size, np(256, N)]
    """

    def __init__(self, config, save_path, device=None):
        super().__init__(config, save_path, device)

        self._image_warping = self._config["data"]["warped_pair"]["enable"]
        self._det_loss_type = self._config["model"]["detector_loss"]["loss_type"]
        self._add_dustbin = True if self._det_loss_type == "softmax" else False
        self._scalar_logs = {}

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
            predictions=outputs["detector"],
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
                predictions=outputs["detector"],
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
                outputs["descriptor"],
                outputs["desc_warp"],
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

        ##### try to minimize the error ######
        add_res_loss = False
        if add_res_loss and iteration % 10 == 0:
            heatmap_org = flattenDetection(outputs["detector"])
            heatmap_org_nms_batch = self.heatmap_to_nms(heatmap_org)
            if self._image_warping:
                heatmap_warp = flattenDetection(outputs["detector_warp"])
                heatmap_warp_nms_batch = self.heatmap_to_nms(heatmap_warp)

            # original: pred
            ## check the loss on given labels!
            outs_res = self.get_residual_loss(
                sample["labels_2D"]
                * torch.tensor(heatmap_org_nms_batch, dtype=torch.float32).unsqueeze(1),
                heatmap_org,
                sample["labels_res"],
            )
            loss_res_ori = (outs_res["loss"] ** 2).mean()
            # warped: pred
            if self._image_warping:
                outs_res_warp = self.get_residual_loss(
                    sample["warped_labels"]
                    * torch.tensor(
                        heatmap_warp_nms_batch, dtype=torch.float32
                    ).unsqueeze(1),
                    heatmap_warp,
                    sample["warped_res"],
                )
                loss_res_warp = (outs_res_warp["loss"] ** 2).mean()
            else:
                loss_res_warp = torch.tensor([0]).to(self._device)
            loss_res = loss_res_ori + loss_res_warp
            # print("loss_res requires_grad: ", loss_res.requires_grad)
            loss += loss_res
            self._scalar_logs.update(
                {"loss_res_ori": loss_res_ori, "loss_res_warp": loss_res_warp}
            )

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

        if iteration % self._config["tensorboard_interval"] == 0 or task == "val":
            logger.info(f"Current iteration: {iteration}")

            heatmap_org_nms_batch = self.heatmap_to_nms(
                flattenDetection(outputs["detector"])
            )
            self._log_random_sample(task, sample, heatmap_org_nms_batch, iteration)

            metrics = self.precision_recall(
                torch.tensor(
                    heatmap_org_nms_batch[:, numpy.newaxis, ...], dtype=torch.float32
                ),
                sample["labels_2D"],
            )
            self._scalar_logs.update(metrics)
            log_losses(self._scalar_logs, task)

        self._tensorboard_logger.log_scalars(iteration, self._scalar_logs, task)

        return loss.item()

    def _forward_step(self, sample, warped_sample: None):
        # TODO: rewrite
        outs = self.net(sample)
        outputs = {"detector": outs["detector"], "descriptor": outs["descriptor"]}

        if self._image_warping:
            assert warped_sample is not None, "Forward step requires warped sample"
            outs_warp = self.net(warped_sample)
            outputs.update(
                {
                    "detector_warp": outs_warp["detector"],
                    "desc_warp": outs_warp["descriptor"],
                }
            )

        return outputs

    def _log_random_sample(
        self,
        task: str,
        sample: dict,
        predictions_heatmap: numpy.ndarray,
        iteration: int,
    ):
        import cv2

        random_sample_index = np.random.randint(0, len(sample["image"]))
        predictions = predictions_heatmap[random_sample_index]
        labels = sample["labels_2D"][random_sample_index].squeeze()
        raw_image = sample["image"][random_sample_index].squeeze().numpy()
        raw_image = 255 - cv2.cvtColor(raw_image, cv2.COLOR_GRAY2RGB)
        predictions_image = raw_image.copy()
        labels_image = raw_image.copy()

        keypoints_coordinates = np.where(predictions > 0.0)
        labels_coordinates = np.where(labels > 0.0)

        self._tensorboard_logger.log_image(task, raw_image, iteration, "raw")
        for y, x in zip(*keypoints_coordinates):
            predictions_image = cv2.circle(
                predictions_image, (x, y), radius=1, color=(255, 0, 0), thickness=-1
            )
        self._tensorboard_logger.log_image(
            task,
            predictions_image,
            iteration,
            "predictions",
        )
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
        for y, x in zip(*keypoints_coordinates):
            labels_image = cv2.circle(
                labels_image, (x, y), radius=1, color=(255, 0, 0), thickness=-1
            )
        self._tensorboard_logger.log_image(
            task,
            labels_image,
            iteration,
            "predictions_labels",
        )

    @staticmethod
    def heatmap_to_nms(heatmap):
        """
        return:
            heatmap_nms_batch: np [batch, H, W]
        """
        return numpy.stack(
            [TrainerHeatmap.heatmap_nms(h) for h in torch_to_numpy(heatmap)], axis=0
        )

    @staticmethod
    def heatmap_nms(heatmap, nms_dist=4, conf_thresh=0.015):
        """
        input:
            heatmap: np [(1), H, W]
        """
        from src.ultrapoint.utils.utils import getPtsFromHeatmap

        heatmap = heatmap.squeeze()
        pts_nms = getPtsFromHeatmap(heatmap, conf_thresh, nms_dist)
        detector_thd_nms_sample = numpy.zeros_like(heatmap)
        detector_thd_nms_sample[
            pts_nms[1, :].astype(int), pts_nms[0, :].astype(int)
        ] = 1

        return detector_thd_nms_sample

    def get_residual_loss(self, labels_2D, heatmap, labels_res):
        if abs(labels_2D).sum() == 0:
            return
        outs_res = self.pred_soft_argmax(
            labels_2D, heatmap, labels_res, patch_size=5, device=self._device
        )
        return outs_res

    @staticmethod
    def pred_soft_argmax(labels_2D, heatmap, labels_res, patch_size, device):
        """

        return:
            dict {'loss': mean of difference btw pred and res}
        """
        label_idx = labels_2D[...].nonzero().long()

        # patch_size = self.config['params']['patch_size']
        patches = extract_patches(
            label_idx.to(device), heatmap.to(device), patch_size=patch_size
        )
        # norm patches
        patches = norm_patches(patches)

        # predict offsets
        patches_log = do_log(patches)
        # soft_argmax
        dxdy = soft_argmax_2d(
            patches_log, normalized_coordinates=False
        )  # tensor [B, N, patch, patch]
        dxdy = dxdy.squeeze(1)  # tensor [N, 2]
        dxdy = dxdy - patch_size // 2

        # extract residual
        def ext_from_points(labels_res, points):
            """
            input:
                labels_res: tensor [batch, channel, H, W]
                points: tensor [N, 4(pos0(batch), pos1(0), pos2(H), pos3(W) )]
            return:
                tensor [N, channel]
            """
            labels_res = labels_res.transpose(1, 2).transpose(2, 3).unsqueeze(1)
            points_res = labels_res[
                points[:, 0], points[:, 1], points[:, 2], points[:, 3], :
            ]  # tensor [N, 2]
            return points_res

        points_res = ext_from_points(labels_res, label_idx)

        return {
            "pred": dxdy,
            "points_res": points_res,
            "loss": dxdy.to(device) - points_res.to(device),
            "patches": patches,
        }
