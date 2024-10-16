import torch
import torch.optim
import torch.utils.data

from typing import Tuple
from torch.nn import BatchNorm2d

from ultrapoint.utils.utils import flattenDetection
from ultrapoint.models.superpoint.unet_parts import Down, InConv
from ultrapoint.utils.torch_helpers import determine_device


class SuperPoint(torch.nn.Module):
    CHANNELS = [64, 64, 128, 128, 256, 256]
    CELL_SIZE = 8
    NUM_GRID_CELLS = CELL_SIZE * CELL_SIZE + 1

    def __init__(self, config):
        super(SuperPoint, self).__init__()
        c1, c2, c3, c4, c5, d1 = self.CHANNELS
        self.inc = InConv(1, c1)
        self.down1 = Down(c1, c2)
        self.down2 = Down(c2, c3)
        self.down3 = Down(c3, c4)
        self.relu = torch.nn.ReLU(inplace=True)

        # Detector Head.
        self.convPa = torch.nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
        self.bnPa = BatchNorm2d(c5)
        self.convPb = torch.nn.Conv2d(
            c5, self.NUM_GRID_CELLS, kernel_size=1, stride=1, padding=0
        )
        self.bnPb = BatchNorm2d(self.NUM_GRID_CELLS)

        # Descriptor Head.
        self.convDa = torch.nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
        self.bnDa = BatchNorm2d(c5)
        self.convDb = torch.nn.Conv2d(c5, d1, kernel_size=1, stride=1, padding=0)
        self.bnDb = BatchNorm2d(d1)

        self._config = config
        self._nms_radius = config["model"]["nms_radius"]
        self._conf_thresh = config["model"]["detection_threshold"]
        self._detection_threshold = config["model"]["detection_threshold"]
        self._remove_borders = config["model"]["remove_borders"]
        self._device = determine_device()
        self.to(self._device)

    def forward_features(self, x: torch.Tensor):
        """
        Forward pass that jointly computes unprocessed point and descriptor
        tensors.
        Input
          x: Image pytorch tensor shaped N x 1 x patch_size x patch_size.
        Output
          detector: Output point pytorch tensor shaped N x NUM_GRID_CELLS x H/CELL_SIZE x W/CELL_SIZE.
          descriptor: Output descriptor pytorch tensor shaped N x 256 x H/CELL_SIZE x W/CELL_SIZE.
        """
        x = x.to(self._device)
        x = self.inc(x)
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)

        # Detector Head.
        detector_output = self.bnPb(self.convPb(self.relu(self.bnPa(self.convPa(x)))))

        # Descriptor Head.
        descriptor_output = self.bnDb(self.convDb(self.relu(self.bnDa(self.convDa(x)))))
        descriptor_output = descriptor_output.div(
            torch.unsqueeze(torch.norm(descriptor_output, p=2, dim=1), 1)
        )  # L2 normalization.

        return {
            "detector_features": detector_output,
            "descriptor_features": descriptor_output,
        }

    def forward(self, image: torch.Tensor):
        output = self.forward_features(image)

        heatmap = flattenDetection(output["detector_features"])
        scores = heatmap[:, 0]
        batch_size, height, width = scores.shape
        if scores.max() < self._conf_thresh:
            return torch.zeros((batch_size, 2, 0)), torch.zeros((batch_size, 0))

        scores = SuperPoint.batched_nms(scores, self._nms_radius)
        scores = SuperPoint.remove_borders(scores, self._remove_borders)

        idxs = torch.where(scores > self._conf_thresh)
        mask = idxs[0] == torch.arange(batch_size, device=scores.device)[:, None]

        # Convert (i, j) to (x, y)
        keypoints_all = torch.stack(idxs[-2:], dim=-1).flip(1).float()
        scores_all = scores[idxs]

        keypoints = []
        scores = []
        descriptors = []
        for i in range(batch_size):
            k = keypoints_all[mask[i]]
            s = scores_all[mask[i]]
            d = SuperPoint.sample_descriptors(
                k[None], output["descriptor_features"][i, None], self.CELL_SIZE
            )
            keypoints.append(k)
            scores.append(s)
            descriptors.append(d.squeeze(0).transpose(0, 1))

        return {
            "keypoints": keypoints,
            "keypoint_scores": scores,
            "descriptors": descriptors,
            "detector_features": output["detector_features"],
            "descriptor_features": output["descriptor_features"],
        }

    @staticmethod
    def normalize_descriptors(descriptors):
        return descriptors.div(torch.unsqueeze(torch.norm(descriptors, p=2, dim=1), 1))

    @staticmethod
    def sort_keypoints(
        keypoints: torch.Tensor, scores: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        sorted_indices = torch.argsort(scores, dim=1, descending=True)
        sorted_keypoints = torch.gather(
            keypoints, 2, sorted_indices.unsqueeze(1).expand(-1, 2, -1)
        )
        sorted_scores = torch.gather(scores, 1, sorted_indices)

        return sorted_keypoints, sorted_scores

    @staticmethod
    def remove_borders(
        scores: torch.Tensor,
        remove_borders: int,
    ) -> torch.Tensor:
        if remove_borders > 0:
            pad = remove_borders
            scores[:, :pad] = -1
            scores[:, :, :pad] = -1
            scores[:, -pad:] = -1
            scores[:, :, -pad:] = -1

        return scores

    @staticmethod
    def sample_descriptors(keypoints, descriptors, s: int = 8):
        """Interpolate descriptors at keypoint locations"""
        b, c, h, w = descriptors.shape
        keypoints = (keypoints + 0.5) / (keypoints.new_tensor([w, h]) * s)
        keypoints = keypoints * 2 - 1  # normalize to (-1, 1)
        descriptors = torch.nn.functional.grid_sample(
            descriptors,
            keypoints.view(b, 1, -1, 2),
            mode="bilinear",
            align_corners=False,
        )
        descriptors = torch.nn.functional.normalize(
            descriptors.reshape(b, c, -1), p=2, dim=1
        )
        return descriptors

    @staticmethod
    def batched_nms(scores, nms_radius: int):
        assert nms_radius >= 0

        def max_pool(x):
            return torch.nn.functional.max_pool2d(
                x, kernel_size=nms_radius * 2 + 1, stride=1, padding=nms_radius
            )

        zeros = torch.zeros_like(scores)
        max_mask = scores == max_pool(scores)
        for _ in range(2):
            supp_mask = max_pool(max_mask.float()) > 0
            supp_scores = torch.where(supp_mask, zeros, scores)
            new_max_mask = supp_scores == max_pool(supp_scores)
            max_mask = max_mask | (new_max_mask & (~supp_mask))
        return torch.where(max_mask, scores, zeros)
