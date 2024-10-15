import torch

from torch.nn import BatchNorm2d
from ultrapoint.models.superpoint.unet_parts import Down, InConv


class SuperPointNet(torch.nn.Module):
    """Pytorch definition of SuperPoint Network."""

    def __init__(self):
        super(SuperPointNet, self).__init__()
        c1, c2, c3, c4, c5, d1 = 64, 64, 128, 128, 256, 256
        det_h = 65
        self.inc = InConv(1, c1)
        self.down1 = Down(c1, c2)
        self.down2 = Down(c2, c3)
        self.down3 = Down(c3, c4)
        self.relu = torch.nn.ReLU(inplace=True)

        # Detector Head.
        self.convPa = torch.nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
        self.bnPa = BatchNorm2d(c5)
        self.convPb = torch.nn.Conv2d(c5, det_h, kernel_size=1, stride=1, padding=0)
        self.bnPb = BatchNorm2d(det_h)

        # Descriptor Head.
        self.convDa = torch.nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
        self.bnDa = BatchNorm2d(c5)
        self.convDb = torch.nn.Conv2d(c5, d1, kernel_size=1, stride=1, padding=0)
        self.bnDb = BatchNorm2d(d1)

    def forward(self, x):
        """
        Forward pass that jointly computes unprocessed point and descriptor
        tensors.
        Input
          x: Image pytorch tensor shaped N x 1 x patch_size x patch_size.
        Output
          detector: Output point pytorch tensor shaped N x 65 x H/8 x W/8.
          descriptor: Output descriptor pytorch tensor shaped N x 256 x H/8 x W/8.
        """
        x = self.inc(x)
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)

        # Detector Head.
        detector = self.bnPb(self.convPb(self.relu(self.bnPa(self.convPa(x)))))

        # Descriptor Head.
        descriptor = self.bnDb(self.convDb(self.relu(self.bnDa(self.convDa(x)))))
        descriptor = descriptor.div(
            torch.unsqueeze(torch.norm(descriptor, p=2, dim=1), 1)
        )  # L2 normalization.

        return {"detector": detector, "descriptor": descriptor}
