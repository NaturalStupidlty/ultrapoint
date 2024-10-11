import torch.nn as nn


def detector_loss(predictions, labels, mask=None, loss_type="softmax"):
    """
    # apply loss on detectors, default is softmax
    :param predictions: prediction
        tensor [batch_size, 65, Hc, Wc]
    :param labels: constructed from labels
        tensor [batch_size, 65, Hc, Wc]
    :param mask: valid region in an image
        tensor [batch_size, 1, Hc, Wc]
    :param loss_type:
        str (l2 or softmax)
        softmax is used in original paper
    :return: normalized loss
        tensor
    """

    if loss_type == "l2":
        loss_func = nn.MSELoss(reduction="mean")
        loss = loss_func(predictions, labels)
    elif loss_type == "softmax":
        loss_func_BCE = nn.BCELoss(reduction="none").cuda()
        loss = loss_func_BCE(nn.functional.softmax(predictions, dim=1), labels)
        loss = (loss.sum(dim=1) * mask).sum()
        loss = loss / (mask.sum() + 1e-10)
    else:
        raise NotImplementedError(f"loss {loss_type} is not implemented")

    return loss
