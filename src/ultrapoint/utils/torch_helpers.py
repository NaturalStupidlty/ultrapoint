import torch
import random
import numpy


def make_deterministic(seed: int = 0):
    """
    Make results deterministic.
    If seed == -1, do not make deterministic.
    Running the script in a deterministic way might slow it down.
    """
    if seed == -1:
        return
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def turn_off_grad():
    """
    Turn off the gradient computation.
    """
    torch.set_grad_enabled(False)


def set_precision(precision: str):
    """
    Set the precision of the matrix multiplication.

    Args:
        precision: The precision of the matrix multiplication. It can be 'highest', 'high' or 'medium'.

    Raises:
        ValueError: If the precision is not 'highest', 'high' or 'medium'.
    """
    if precision not in ["highest", "high", "medium"]:
        raise ValueError("The precision must be 'highest', 'high' or 'medium'.")

    torch.set_float32_matmul_precision(precision)


def determine_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
