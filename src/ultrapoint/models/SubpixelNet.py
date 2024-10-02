"""Old version of SuperpointNet. Use it together with 
logs/magicpoint_synth20/checkpoints/superPointNet_200000_checkpoint.pth.tar

"""

import torch
from torch.nn import BatchNorm2d
from src.ultrapoint.models.unet_parts import InConv, Down, Up, OutConv


class SubpixelNet(torch.nn.Module):
    """Pytorch definition of SuperPoint Network."""

    def __init__(self, subpixel_channel=1):
        super(SubpixelNet, self).__init__()
        c1, c2, c3, c4, c5, d1 = 64, 64, 128, 128, 256, 256
        det_h = 65
        self.inc = InConv(1, c1)
        self.down1 = Down(c1, c2)
        self.down2 = Down(c2, c3)
        self.down3 = Down(c3, c4)
        # self.down4 = down(c4, 512)
        self.up1 = Up(c4 + c3, c2)
        self.up2 = Up(c2 + c2, c1)
        self.up3 = Up(c1 + c1, c1)
        self.outc = OutConv(c1, subpixel_channel)
        self.relu = torch.nn.ReLU(inplace=True)
        # self.outc = OutConv(64, n_classes)
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

    @staticmethod
    def soft_argmax_2d(patches):
        """
        params:
            patches: (B, N, H, W)
        return:
            coor: (B, N, 2)  (x, y)

        """
        import torchgeometry as tgm

        m = tgm.contrib.SpatialSoftArgmax2d()
        coords = m(patches)  # 1x4x2
        return coords

    def forward(self, x, subpixel=False):
        """Forward pass that jointly computes unprocessed point and descriptor
        tensors.
        Input
          x: Image pytorch tensor shaped N x 1 x patch_size x patch_size.
        Output
          semi: Output point pytorch tensor shaped N x 65 x H/8 x W/8.
          desc: Output descriptor pytorch tensor shaped N x 256 x H/8 x W/8.
        """
        # Let's stick to this version: first BN, then relu
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        # x5 = self.down4(x4)

        cPa = self.bnPa(self.relu(self.convPa(x4)))
        semi = self.bnPb(self.convPb(cPa))
        # Descriptor Head.
        cDa = self.bnDa(self.relu(self.convDa(x4)))
        desc = self.bnDb(self.convDb(cDa))

        dn = torch.norm(desc, p=2, dim=1)  # Compute the norm.
        desc = desc.div(torch.unsqueeze(dn, 1))  # Divide by norm to normalize.

        # subpixel = True
        if subpixel:
            x = self.up1(x4, x3)
            x = self.up2(x, x2)
            x = self.up3(x, x1)
            x = self.outc(x)
            # print("x: ", x.shape)
            return semi, desc, x

        return semi, desc
