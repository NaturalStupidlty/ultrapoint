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
            self._writer.add_image(f"{task}/{name}/{index}", image, iteration)
        else:
            self._writer.add_image(f"{task}/{name}", image, iteration)

    def log_scalars(self, iteration: int, scalars: dict, task: str = "training"):
        for name, scalar in scalars.items():
            self._writer.add_scalar(f"{task}/{name}", scalar, iteration)
