import torch
import torch.nn.functional as F


def sample_descriptors(keypoints: torch.Tensor, descriptors: torch.Tensor, s: int = 8):
    """Interpolate descriptors at keypoint locations (unchanged)."""
    b, c, h, w = descriptors.shape
    keypoints = (keypoints + 0.5) / (keypoints.new_tensor([w, h]) * s)
    keypoints = keypoints * 2 - 1  # normalize to (-1, 1)
    descriptors = F.grid_sample(
        descriptors, keypoints.view(b, 1, -1, 2), mode="bilinear", align_corners=False
    )
    descriptors = F.normalize(descriptors.reshape(b, c, -1), p=2, dim=1)
    return descriptors


def batched_nms(scores: torch.Tensor, nms_radius: int):
    """Fast approximate non‑maximum‑suppression identical to the reference impl."""
    assert nms_radius >= 0

    def _max_pool(x):
        return F.max_pool2d(x, kernel_size=nms_radius * 2 + 1, stride=1, padding=nms_radius)

    zeros = torch.zeros_like(scores)
    max_mask = scores == _max_pool(scores)
    for _ in range(2):
        supp_mask = _max_pool(max_mask.float()) > 0
        supp_scores = torch.where(supp_mask, zeros, scores)
        new_max_mask = supp_scores == _max_pool(supp_scores)
        max_mask = max_mask | (new_max_mask & (~supp_mask))
    return torch.where(max_mask, scores, zeros)


def select_top_k_keypoints(keypoints: torch.Tensor, scores: torch.Tensor, k: int):
    if k >= len(keypoints):
        return keypoints, scores
    scores, indices = torch.topk(scores, k, dim=0, sorted=True)
    return keypoints[indices], scores
