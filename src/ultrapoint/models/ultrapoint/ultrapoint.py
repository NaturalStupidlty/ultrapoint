import torch.nn as nn
from torch.nn import BatchNorm2d

from ultrapoint.models.ultrapoint.model_blocks import (
    Down,
    InConv,
    SeparableConv2d,
)
from ultrapoint.models.superpoint.superpoint import SuperPoint as SuperPointBase
from ultrapoint.utils.torch_helpers import determine_device


class UltraPoint(SuperPointBase):
    def __init__(self, **config):
        super(UltraPoint, self).__init__(**config)
        c1, c2, c3, c4, c5, d1 = self.CHANNELS

        self.inc = InConv(1, c1)
        self.down1 = Down(c1, c2)
        self.down2 = Down(c2, c3)
        self.down3 = Down(c3, c4)
        self.relu = nn.ReLU(inplace=True)

        # Detector Head using separable conv
        self.convPa = SeparableConv2d(c4, c5, kernel_size=3, padding=1)
        self.bnPa = BatchNorm2d(c5)
        self.convPb = nn.Conv2d(
            c5, self.NUM_GRID_CELLS, kernel_size=1, stride=1, padding=0
        )
        self.bnPb = BatchNorm2d(self.NUM_GRID_CELLS)

        # Descriptor Head using separable conv
        self.convDa = SeparableConv2d(c4, c5, kernel_size=3, padding=1)
        self.bnDa = BatchNorm2d(c5)
        self.convDb = nn.Conv2d(
            c5, d1, kernel_size=1, stride=1, padding=0
        )
        self.bnDb = BatchNorm2d(d1)
        self._device = determine_device()
        self.to(self._device)
