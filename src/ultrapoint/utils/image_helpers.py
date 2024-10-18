from PIL import Image
import numpy as np
from typing import Tuple


def read_image(
    path: str,
    resize: Tuple[int, int] = None,
    grayscale: bool = True,
    normalized: bool = True,
):
    """
    Read an image from a file, resize it, and normalize if necessary.

    Args:
        path (str): Path to the image file.
        resize (Tuple[int, int]): Height and width to resize the image to.
        grayscale (bool): Whether to read the image in grayscale or not.
        normalized (bool): Whether to normalize the image or not.

    Returns:
        np.ndarray: Processed image as a NumPy array.
    """
    image = Image.open(path)

    if grayscale:
        image = image.convert("L")

    if resize:
        image.thumbnail(resize, Image.Resampling.LANCZOS)

    image_np = np.array(image)

    if normalized:
        return image_np.astype("float32") / 255.0
    else:
        return image_np
