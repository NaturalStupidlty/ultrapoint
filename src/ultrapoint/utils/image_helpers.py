import cv2
from typing import Tuple


def read_image(
    path,
    resize: Tuple[int, int] = None,
    keep_ratio: bool = False,
    grayscale: bool = True,
    normalized: bool = True,
):
    """
    Read an image from a file and resize it and normalizes if necessary.

    Args:
        path (str): Path to the image file.
        resize (Tuple[int, int]): Height and width to resize the image to.
        keep_ratio (bool): Whether to keep the aspect ratio of the image or not
        grayscale (bool): Whether to read the image in grayscale or not.
        normalized (bool): Whether to normalize the image or not.
    """
    image = cv2.imread(path)
    if keep_ratio:
        scale_factor = max(resize[0] / image.shape[0], resize[1] / image.shape[1])
        image = image[: int(resize[0] / scale_factor), : int(resize[1] / scale_factor)]
    image = cv2.resize(
        image,
        (resize[1], resize[0]),
        interpolation=cv2.INTER_AREA,
    )
    if grayscale:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    if normalized:
        return image.astype("float32") / 255.0
    else:
        return image
