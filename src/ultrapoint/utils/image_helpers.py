import cv2

from typing import Tuple


def read_image(path, resize: Tuple[int, int] = None, normalized: bool = True):
    """
    Read an image from a file and resize it and normalizes if necessary.

    Args:
        path (str): Path to the image file.
        resize (Tuple[int, int]): Height and width to resize the image to.
        normalized (bool): Whether to normalize the image or not.
    """
    input_image = cv2.imread(path)
    input_image = cv2.resize(
        input_image,
        (resize[1], resize[0]),
        interpolation=cv2.INTER_AREA,
    )
    input_image = cv2.cvtColor(input_image, cv2.COLOR_RGB2GRAY)

    if normalized:
        return input_image.astype("float32") / 255.0
    else:
        return input_image
