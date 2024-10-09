import torch
import numpy as np

from tensorboardX import SummaryWriter
from loguru import logger

from ultrapoint.utils.torch_helpers import to_numpy
from ultrapoint.utils.utils import saveImg
from ultrapoint.utils.utils import precisionRecall_torch
from ultrapoint.utils.utils import (
    flattenDetection,
)


def thd_img(img, thd=0.015):
    """
    thresholding the image.
    :param img:
    :param thd:
    :return:
    """
    img[img < thd] = 0
    img[img >= thd] = 1

    return img


def img_overlap(img_r, img_g, img_gray):  # img_b repeat
    img = np.concatenate((img_gray, img_gray, img_gray), axis=0)
    img[0, :, :] += img_r[0, :, :]
    img[1, :, :] += img_g[0, :, :]
    img[img > 1] = 1
    img[img < 0] = 0

    return img


class TensorboardLogger:
    def __init__(self, save_path: str, config: dict):
        self._config = config
        self._writer = SummaryWriter(save_path)

    def close(self):
        self._writer.close()

    def log_images(
        self, task: str, image: torch.Tensor, iteration: int, name: str = "img"
    ) -> None:
        """
        Add images to TensorBoard for visualization.

        If the input image tensor has four dimensions (a batch of images),
        it logs up to the first five images. If it's a single image tensor,
        it logs that image directly.

        :param task: The task or experiment name used to group the images in TensorBoard.
        :param image: A torch.Tensor containing the image(s) to log.
                      Shape can be either (N, C, H, W) for a batch of images
                      or (C, H, W) for a single image.
        :param iteration: The current iteration number for logging, typically corresponds to the training step.
        :param name: An optional name for the image category, default is "img".
        """
        if image.dim() == 4:
            for i in range(min(image.shape[0], 5)):
                self.log_image(task, image[i, :, :, :], iteration, name, i)
        else:
            self.log_image(task, image, iteration, name)

    def log_image(
        self,
        task: str,
        image: torch.Tensor,
        iteration: int,
        name: str,
        index: int = None,
    ) -> None:
        """
        Log a single image to TensorBoard.

        This method is called by `log_images` to handle the logging of
        individual images, allowing for cleaner code and reduced duplication.

        :param task: The task or experiment name used to identify the image in TensorBoard.
        :param image: A torch.Tensor containing the image to log.
                      Expected shape is (C, H, W).
        :param iteration: The current iteration number for logging.
        :param name: The name for the image category.
        :param index: An optional index for naming the logged image when logging a batch.
                      If provided, it will be included in the TensorBoard path.
        """
        if index is not None:
            self._writer.add_image(f"{task}-{name}/{index}", image, iteration)
        else:
            self._writer.add_image(f"{task}-{name}", image, iteration)

    # tensorboard
    def addImg2tensorboard(
        self,
        iteration,
        img,
        labels_2D,
        semi,
        img_warp=None,
        labels_warp_2D=None,
        mask_warp_2D=None,
        semi_warp=None,
        mask_3D_flattened=None,
        task="training",
    ):
        """
        # deprecated: add images to tensorboard
        :param img:
        :param labels_2D:
        :param semi:
        :param img_warp:
        :param labels_warp_2D:
        :param mask_warp_2D:
        :param semi_warp:
        :param mask_3D_flattened:
        :param task:
        :return:
        """
        logger.info("Adding images to tensorboard")

        semi_flat = flattenDetection(semi[0, :, :, :])
        semi_warp_flat = flattenDetection(semi_warp[0, :, :, :])

        thd = self._config["model"]["detection_threshold"]
        semi_thd = thd_img(semi_flat, thd=thd)
        semi_warp_thd = thd_img(semi_warp_flat, thd=thd)

        result_overlap = img_overlap(
            to_numpy(labels_2D[0, :, :, :]),
            to_numpy(semi_thd),
            to_numpy(img[0, :, :, :]),
        )

        self._writer.add_image(
            task + "-detector_output_thd_overlay", result_overlap, iteration
        )
        saveImg(
            result_overlap.transpose([1, 2, 0])[..., [2, 1, 0]] * 255, "test_0.png"
        )  # rgb to bgr * 255

        result_overlap = img_overlap(
            to_numpy(labels_warp_2D[0, :, :, :]),
            to_numpy(semi_warp_thd),
            to_numpy(img_warp[0, :, :, :]),
        )
        self._writer.add_image(
            task + "-warp_detector_output_thd_overlay", result_overlap, iteration
        )
        saveImg(
            result_overlap.transpose([1, 2, 0])[..., [2, 1, 0]] * 255, "test_1.png"
        )  # rgb to bgr * 255

        mask_overlap = img_overlap(
            to_numpy(1 - mask_warp_2D[0, :, :, :]) / 2,
            np.zeros_like(to_numpy(img_warp[0, :, :, :])),
            to_numpy(img_warp[0, :, :, :]),
        )

        for i in range(self._config["model"]["batch_size"]):
            if i < 5:
                self._writer.add_image(
                    task + "-mask_warp_origin", mask_warp_2D[i, :, :, :], iteration
                )
                self._writer.add_image(
                    task + "-mask_warp_3D_flattened",
                    mask_3D_flattened[i, :, :],
                    iteration,
                )
        self._writer.add_image(task + "-mask_warp_overlay", mask_overlap, iteration)

    def tb_scalar_dict(self, iteration, losses, task="training"):
        """
        # add scalar dictionary to tensorboard
        :param losses:
        :param task:
        :return:
        """
        for element in list(losses):
            self._writer.add_scalar(task + "-" + element, losses[element], iteration)

    def tb_images_dict(self, iteration, task, tb_imgs, max_img=5):
        """
        # add image dictionary to tensorboard
        :param task:
            str (train, val)
        :param tb_imgs:
        :param max_img:
            int - number of images
        :return:
        """
        for element in list(tb_imgs):
            for idx in range(tb_imgs[element].shape[0]):
                if idx >= max_img:
                    break
                # print(f"element: {element}")
                self._writer.add_image(
                    task + "-" + element + "/%d" % idx,
                    tb_imgs[element][idx, ...],
                    iteration,
                )

    def tb_hist_dict(self, iteration, task, tb_dict):
        for element in list(tb_dict):
            self._writer.add_histogram(
                task + "-" + element, tb_dict[element], iteration
            )

    def add2tensorboard_nms(
        self, iteration, img, labels_2D, semi, task="training", batch_size=1
    ):
        """
        # deprecated:
        :param img:
        :param labels_2D:
        :param semi:
        :param task:
        :param batch_size:
        :return:
        """
        from src.ultrapoint.utils.utils import getPtsFromHeatmap
        from src.ultrapoint.utils.utils import box_nms

        boxNms = False

        nms_dist = self._config["model"]["nms"]
        conf_thresh = self._config["model"]["detection_threshold"]

        precision_recall_list = []
        precision_recall_boxnms_list = []
        for idx in range(batch_size):
            semi_flat_tensor = flattenDetection(semi[idx, :, :, :]).detach()
            semi_flat = to_numpy(semi_flat_tensor)
            semi_thd = np.squeeze(semi_flat, 0)
            pts_nms = getPtsFromHeatmap(semi_thd, conf_thresh, nms_dist)
            semi_thd_nms_sample = np.zeros_like(semi_thd)
            semi_thd_nms_sample[
                pts_nms[1, :].astype(np.int), pts_nms[0, :].astype(np.int)
            ] = 1

            label_sample = torch.squeeze(labels_2D[idx, :, :, :])
            # pts_nms = getPtsFromHeatmap(label_sample.numpy(), conf_thresh, nms_dist)
            # label_sample_rms_sample = np.zeros_like(label_sample.numpy())
            # label_sample_rms_sample[pts_nms[1, :].astype(np.int), pts_nms[0, :].astype(np.int)] = 1
            label_sample_nms_sample = label_sample

            if idx < 5:
                result_overlap = img_overlap(
                    np.expand_dims(label_sample_nms_sample, 0),
                    np.expand_dims(semi_thd_nms_sample, 0),
                    to_numpy(img[idx, :, :, :]),
                )
                self._writer.add_image(
                    task + "-detector_output_thd_overlay-NMS" + "/%d" % idx,
                    result_overlap,
                    iteration,
                )
            assert semi_thd_nms_sample.shape == label_sample_nms_sample.size()
            precision_recall = precisionRecall_torch(
                torch.from_numpy(semi_thd_nms_sample), label_sample_nms_sample
            )
            precision_recall_list.append(precision_recall)

            if boxNms:
                semi_flat_tensor_nms = box_nms(
                    semi_flat_tensor.squeeze(), nms_dist, min_prob=conf_thresh
                ).cpu()
                semi_flat_tensor_nms = (semi_flat_tensor_nms >= conf_thresh).float()

                if idx < 5:
                    result_overlap = img_overlap(
                        np.expand_dims(label_sample_nms_sample, 0),
                        semi_flat_tensor_nms.numpy()[np.newaxis, :, :],
                        to_numpy(img[idx, :, :, :]),
                    )
                    self._writer.add_image(
                        task + "-detector_output_thd_overlay-boxNMS" + "/%d" % idx,
                        result_overlap,
                        iteration,
                    )
                precision_recall_boxnms = precisionRecall_torch(
                    semi_flat_tensor_nms, label_sample_nms_sample
                )
                precision_recall_boxnms_list.append(precision_recall_boxnms)

        precision = np.mean(
            [
                precision_recall["precision"]
                for precision_recall in precision_recall_list
            ]
        )
        recall = np.mean(
            [precision_recall["recall"] for precision_recall in precision_recall_list]
        )
        self._writer.add_scalar(task + "-precision_nms", precision, iteration)
        self._writer.add_scalar(task + "-recall_nms", recall, iteration)
        logger.info(
            "-- [%s-%d-fast NMS] Precision: %.4f, Recall: %.4f"
            % (task, iteration, precision, recall)
        )
        if boxNms:
            precision = np.mean(
                [
                    precision_recall["precision"]
                    for precision_recall in precision_recall_boxnms_list
                ]
            )
            recall = np.mean(
                [
                    precision_recall["recall"]
                    for precision_recall in precision_recall_boxnms_list
                ]
            )
            self._writer.add_scalar(task + "-precision_boxnms", precision, iteration)
            self._writer.add_scalar(task + "-recall_boxnms", recall, iteration)
            logger.info(
                "-- [%s-%d-boxNMS] Precision: %.4f, Recall: %.4f"
                % (task, iteration, precision, recall)
            )
