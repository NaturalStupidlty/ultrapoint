from types import SimpleNamespace
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import InvertedResidual, DWConvBlock
from ultrapoint.models.superpoint.common import (
    sample_descriptors,
    batched_nms,
    select_top_k_keypoints,
)


class UltraPoint(nn.Module):
    """SuperPoint with efficient backbone.

    The public API is identical to the reference implementation.
    """

    default_conf = {
        "nms_radius": 4,
        "max_num_keypoints": None,
        "detection_threshold": 0.005,
        "remove_borders": 4,
        "descriptor_dim": 256,
        # Channel configuration for each down‑sampling stage (excluding the input)
        "channels": [16, 32, 64, 128],
        # MobileNetV2 expansion factors for each stage
        "expansion": [1, 4, 4, 4],
    }

    def __init__(self, **conf):
        super().__init__()
        conf = {**self.default_conf, **conf}
        self.conf = SimpleNamespace(**conf)

        # ------------------------------------------------------------------
        # Backbone encoder (MobileNetV2‑like)
        # ------------------------------------------------------------------
        # First layer: standard DW separable conv stride 1 (we keep 1 px resolution)
        layers: List[nn.Module] = [DWConvBlock(1, self.conf.channels[0], stride=1)]

        # Subsequent stages: [InvertedResidual × k] + downsample by stride 2 once per stage
        c_prev = self.conf.channels[0]
        for i, c_out in enumerate(self.conf.channels[1:]):
            # one stride‑2 IR block for downsample
            layers.append(InvertedResidual(c_prev, c_out, stride=2, expansion=self.conf.expansion[i + 1]))
            # followed by one stride‑1 IR block to enrich features
            layers.append(InvertedResidual(c_out, c_out, stride=1, expansion=self.conf.expansion[i + 1]))
            c_prev = c_out
        self.backbone = nn.Sequential(*layers)

        # Total downsampling stride = 2^(num_downsamples) -> len(channels)-1
        self.stride = 2 ** (len(self.conf.channels) - 1)

        # ------------------------------------------------------------------
        # Detector and Descriptor heads (unchanged in principle)
        # ------------------------------------------------------------------
        head_channels = max(64, self.conf.channels[-1])  # ensure sufficient capacity
        self.detector = nn.Sequential(
            DWConvBlock(self.conf.channels[-1], head_channels, stride=1),
            # Output channels = stride^2 + 1 (same as original design)
            nn.Conv2d(head_channels, self.stride ** 2 + 1, kernel_size=1, bias=True),
        )
        self.descriptor = nn.Sequential(
            DWConvBlock(self.conf.channels[-1], head_channels, stride=1),
            nn.Conv2d(head_channels, self.conf.descriptor_dim, kernel_size=1, bias=True),
        )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, image: torch.Tensor):
        """Run SuperPoint extraction on grayscale or RGB images.

        Args:
            image (Tensor): shape (B, 1 or 3, H, W)
        Returns:
            dict with keys [keypoints, keypoint_scores, descriptors, detector_features, descriptor_features]
        """
        if image.shape[1] == 3:  # RGB → gray
            scale = image.new_tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)
            image = (image * scale).sum(1, keepdim=True)

        features = self.backbone(image)
        detector_features = self.detector(features)
        descriptor_features = self.descriptor(features)
        descriptors_dense = F.normalize(descriptor_features, p=2, dim=1)

        # --------------------------------------------------------------
        # Decode detector logits → score heatmap (same logic as ref impl)
        # --------------------------------------------------------------
        scores = F.softmax(detector_features, 1)[:, :-1]
        b, _, h, w = scores.shape
        scores = scores.permute(0, 2, 3, 1).reshape(b, h, w, self.stride, self.stride)
        scores = scores.permute(0, 1, 3, 2, 4).reshape(b, h * self.stride, w * self.stride)
        scores = batched_nms(scores, self.conf.nms_radius)

        # Remove border keypoints if requested
        if self.conf.remove_borders:
            pad = self.conf.remove_borders
            scores[:, :pad] = -1
            scores[:, :, :pad] = -1
            scores[:, -pad:] = -1
            scores[:, :, -pad:] = -1

        # --------------------------------------------------------------
        # Extract keypoints
        # --------------------------------------------------------------
        if b > 1:
            idxs = torch.where(scores > self.conf.detection_threshold)
            batch_mask = idxs[0] == torch.arange(b, device=scores.device)[:, None]
        else:
            scores = scores.squeeze(0)
            idxs = torch.where(scores > self.conf.detection_threshold)

        keypoints_all = torch.stack(idxs[-2:], dim=-1).flip(1).float()  # (N,2) xy
        scores_all = scores[idxs]

        keypoints: List[torch.Tensor] = []
        scores_list: List[torch.Tensor] = []
        descriptors: List[torch.Tensor] = []
        for i in range(b):
            if b > 1:
                k = keypoints_all[batch_mask[i]]
                s = scores_all[batch_mask[i]]
            else:
                k = keypoints_all
                s = scores_all
            if self.conf.max_num_keypoints is not None:
                k, s = select_top_k_keypoints(k, s, self.conf.max_num_keypoints)
            d = sample_descriptors(k[None], descriptors_dense[i, None], self.stride)
            keypoints.append(k)
            scores_list.append(s)
            descriptors.append(d.squeeze(0).transpose(0, 1))

        return {
            "keypoints": keypoints,
            "keypoint_scores": scores_list,
            "descriptors": descriptors,
            "detector_features": detector_features,
            "descriptor_features": descriptor_features,
        }
