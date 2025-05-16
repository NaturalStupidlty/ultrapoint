"""PyTorch implementation of the SuperPoint model,
   derived from the TensorFlow re-implementation (2018).
   Authors: Rémi Pautrat, Paul-Edouard Sarlin
   Adapted by: Ihor Olkhovatyi
"""
from collections import OrderedDict
from types import SimpleNamespace

import torch.nn as nn
import torch

from .common import sample_descriptors, batched_nms, select_top_k_keypoints


class VGGBlock(nn.Sequential):
    def __init__(self, c_in, c_out, kernel_size, relu=True):
        padding = (kernel_size - 1) // 2
        conv = nn.Conv2d(
            c_in, c_out, kernel_size=kernel_size, stride=1, padding=padding
        )
        activation = nn.ReLU(inplace=True) if relu else nn.Identity()
        bn = nn.BatchNorm2d(c_out, eps=0.001)
        super().__init__(
            OrderedDict(
                [
                    ("conv", conv),
                    ("activation", activation),
                    ("bn", bn),
                ]
            )
        )


class SuperPoint(nn.Module):
    default_conf = {
        "nms_radius": 4,
        "max_num_keypoints": None,
        "detection_threshold": 0.005,
        "remove_borders": 4,
        "descriptor_dim": 256,
        "channels": [64, 64, 128, 128, 256],
    }

    def __init__(self, **conf):
        super().__init__()
        conf = {**self.default_conf, **conf}
        self.conf = SimpleNamespace(**conf)
        self.stride = 2 ** (len(self.conf.channels) - 2)
        channels = [1, *self.conf.channels[:-1]]

        backbone = []
        for i, c in enumerate(channels[1:], 1):
            layers = [VGGBlock(channels[i - 1], c, 3), VGGBlock(c, c, 3)]
            if i < len(channels) - 1:
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            backbone.append(nn.Sequential(*layers))
        self.backbone = nn.Sequential(*backbone)

        c = self.conf.channels[-1]
        self.detector = nn.Sequential(
            VGGBlock(channels[-1], c, 3),
            VGGBlock(c, self.stride**2 + 1, 1, relu=False),
        )
        self.descriptor = nn.Sequential(
            VGGBlock(channels[-1], c, 3),
            VGGBlock(c, self.conf.descriptor_dim, 1, relu=False),
        )

    def forward(self, image):
        if image.shape[1] == 3:  # RGB to gray
            scale = image.new_tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)
            image = (image * scale).sum(1, keepdim=True)

        features = self.backbone(image)
        descriptor_features = self.descriptor(features)
        descriptors_dense = torch.nn.functional.normalize(
            descriptor_features, p=2, dim=1
        )

        # Decode the detection scores
        detector_features = self.detector(features)
        scores = torch.nn.functional.softmax(detector_features, 1)[:, :-1]
        b, _, h, w = scores.shape
        scores = scores.permute(0, 2, 3, 1).reshape(b, h, w, self.stride, self.stride)
        scores = scores.permute(0, 1, 3, 2, 4).reshape(
            b, h * self.stride, w * self.stride
        )
        scores = batched_nms(scores, self.conf.nms_radius)

        # Discard keypoints near the image borders
        if self.conf.remove_borders:
            pad = self.conf.remove_borders
            scores[:, :pad] = -1
            scores[:, :, :pad] = -1
            scores[:, -pad:] = -1
            scores[:, :, -pad:] = -1

        # Extract keypoints
        if b > 1:
            idxs = torch.where(scores > self.conf.detection_threshold)
            mask = idxs[0] == torch.arange(b, device=scores.device)[:, None]
        else:  # Faster shortcut
            scores = scores.squeeze(0)
            idxs = torch.where(scores > self.conf.detection_threshold)

        # Convert (i, j) to (x, y)
        keypoints_all = torch.stack(idxs[-2:], dim=-1).flip(1).float()
        scores_all = scores[idxs]

        keypoints = []
        scores = []
        descriptors = []
        for i in range(b):
            if b > 1:
                k = keypoints_all[mask[i]]
                s = scores_all[mask[i]]
            else:
                k = keypoints_all
                s = scores_all
            if self.conf.max_num_keypoints is not None:
                k, s = select_top_k_keypoints(k, s, self.conf.max_num_keypoints)
            d = sample_descriptors(k[None], descriptors_dense[i, None], self.stride)
            keypoints.append(k)
            scores.append(s)
            descriptors.append(d.squeeze(0).transpose(0, 1))

        return {
            "keypoints": keypoints,
            "keypoint_scores": scores,
            "descriptors": descriptors,
            "detector_features": detector_features,
            "descriptor_features": descriptor_features,
        }
