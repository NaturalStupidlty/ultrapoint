import torch

from tensorboardX import SummaryWriter


class TensorboardLogger:
    def __init__(self, save_path: str, config: dict):
        self._config = config
        self._writer = SummaryWriter(save_path)

    def close(self):
        self._writer.close()

    def log_image(
        self,
        task: str,
        image: torch.Tensor,
        iteration: int,
        name: str,
        group: str = "images",
        normalize: bool = True,
        dataformats: str = "HWC",
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
        :param group: The group name for the image category.
                      Default is "images".
        :param normalize: Whether to normalize the image tensor before logging.
                          Default is True.
        :param dataformats: The format of the input image tensor.
                            Default is "HWC" (Height, Width, Channels).
        """
        if normalize:
            image = image / 255.0
        self._writer.add_image(
            f"{task}/{group}/{name}", image, iteration, dataformats=dataformats
        )

    def log_scalars(
        self,
        iteration: int,
        scalars: dict,
        task: str = "training",
        group: str = "metrics",
    ) -> None:
        for name, scalar in scalars.items():
            self._writer.add_scalar(f"{task}/{group}/{name}", scalar, iteration)
