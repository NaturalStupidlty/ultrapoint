from collections import OrderedDict
from typing import List

import torch.nn as nn


class DWConvBlock(nn.Sequential):
    """Depth‑wise separable convolution: DW 3×3 + PW 1×1."""

    def __init__(self, c_in: int, c_out: int, stride: int = 1):
        padding = 1
        super().__init__(
            OrderedDict(
                [
                    ("dw", nn.Conv2d(c_in, c_in, 3, stride=stride, padding=padding, groups=c_in, bias=False)),
                    ("dw_bn", nn.BatchNorm2d(c_in, eps=1e-3)),
                    ("dw_relu", nn.ReLU(inplace=True)),
                    ("pw", nn.Conv2d(c_in, c_out, 1, stride=1, bias=False)),
                    ("pw_bn", nn.BatchNorm2d(c_out, eps=1e-3)),
                    ("pw_relu", nn.ReLU(inplace=True)),
                ]
            )
        )


class InvertedResidual(nn.Module):
    """MobileNetV2 inverted residual with linear bottleneck."""

    def __init__(self, c_in: int, c_out: int, stride: int, expansion: int):
        super().__init__()
        self.use_res_connect = stride == 1 and c_in == c_out
        hidden_dim = c_in * expansion
        layers: List[nn.Module] = []
        # expansion (1×1 conv)
        if expansion != 1:
            layers.append(nn.Conv2d(c_in, hidden_dim, 1, bias=False))
            layers.append(nn.BatchNorm2d(hidden_dim, eps=1e-3))
            layers.append(nn.ReLU6(inplace=True))
        # depthwise 3×3
        layers.append(nn.Conv2d(hidden_dim, hidden_dim, 3, stride=stride, padding=1, groups=hidden_dim, bias=False))
        layers.append(nn.BatchNorm2d(hidden_dim, eps=1e-3))
        layers.append(nn.ReLU6(inplace=True))
        # projection
        layers.append(nn.Conv2d(hidden_dim, c_out, 1, bias=False))
        layers.append(nn.BatchNorm2d(c_out, eps=1e-3))
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        return self.conv(x)
