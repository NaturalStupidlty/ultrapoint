import torch
import torch.optim
import torch.utils.data

import torch.nn as nn
import numpy as np

from loguru import logger
from ultrapoint.models.models_factory import ModelsFactory
from ultrapoint.utils.utils import flattenDetection
from ultrapoint.utils.torch_helpers import (
    determine_device,
    clear_memory,
    set_precision,
    make_deterministic,
)


class SuperPointFrontend:
    def __init__(self, config, train: bool = False):
        self._config = config
        self._batch_size = config["model"]["batch_size"]
        self._nms_dist = config["model"]["nms_radius"]
        self._conf_thresh = config["model"]["detection_threshold"]
        self._nn_thresh = config["model"]["nn_thresh"]
        self._remove_borders = config["model"]["remove_borders"]
        self._train = train
        self._cell_size = 8

        clear_memory()
        make_deterministic(config["seed"])
        set_precision(config["precision"])
        self._device = determine_device()
        logger.info(f"Using device: {self._device}")

        self._model = ModelsFactory.create(
            model_name=self._config["model"]["name"],
            state=torch.load(config["pretrained"])["model_state_dict"],
            **self._config["model"]["params"],
        ).to(self._device)
        self._model = nn.DataParallel(self._model)
        if self._train:
            self._model.train()
        else:
            self._model.eval()

    def __call__(self, image: torch.Tensor):
        image = image.to(self._device)
        if self._train:
            output = self._model.forward(image)
        else:
            # TODO: do we need this?
            with torch.no_grad():
                output = self._model.forward(image)

        # flatten detection
        heatmap = flattenDetection(output["detector"]).cpu().detach().numpy()

        # extract keypoints
        keypoints = [
            self.heatmap_to_keypoints(
                heatmap[i, :, :, :],
                self._conf_thresh,
                self._nms_dist,
                self._remove_borders,
            )
            for i in range(self._batch_size)
        ]

        descriptors = nn.functional.interpolate(
            output["descriptor"],
            scale_factor=(self._cell_size, self._cell_size),
            mode="bilinear",
        )
        descriptors = (
            SuperPointFrontend.normalize_descriptors(descriptors).cpu().detach().numpy()
        )
        descriptors = [
            descriptors[
                i, :, keypoints[i][1, :].astype(int), keypoints[i][0, :].astype(int)
            ].transpose()
            for i in range(len(keypoints))
        ]

        return keypoints, descriptors

    @staticmethod
    def normalize_descriptors(descriptors):
        return descriptors.div(torch.unsqueeze(torch.norm(descriptors, p=2, dim=1), 1))

    @staticmethod
    def heatmap_to_keypoints(
        heatmap, confidence_threshold: float, nms_dist: float, remove_borders: int
    ):
        heatmap = heatmap.squeeze()
        H, W = heatmap.shape[0], heatmap.shape[1]
        xs, ys = np.where(heatmap >= confidence_threshold)
        if len(xs) == 0:
            return np.zeros((3, 0))
        pts = np.zeros((3, len(xs)))  # Populate point data sized 3xN.
        pts[0, :] = ys  # abuse of ys, xs
        pts[1, :] = xs
        pts[2, :] = heatmap[xs, ys]  # check the (x, y) here
        pts, _ = SuperPointFrontend.nms_fast(
            pts, H, W, dist_thresh=nms_dist
        )  # Apply NMS.
        inds = np.argsort(pts[2, :])
        pts = pts[:, inds[::-1]]  # Sort by confidence.

        # Remove points along border.
        toremoveW = np.logical_or(
            pts[0, :] < remove_borders, pts[0, :] >= (W - remove_borders)
        )
        toremoveH = np.logical_or(
            pts[1, :] < remove_borders, pts[1, :] >= (H - remove_borders)
        )
        toremove = np.logical_or(toremoveW, toremoveH)

        return pts[:, ~toremove]

    @staticmethod
    def nms_fast(in_corners, H, W, dist_thresh):
        """
        Run a faster approximate Non-Max-Suppression on numpy corners shaped:
          3xN [x_i,y_i,conf_i]^T

        Algo summary: Create a grid sized HxW. Assign each corner location a 1, rest
        are zeros. Iterate through all the 1's and convert them either to -1 or 0.
        Suppress points by setting nearby values to 0.

        Grid Value Legend:
        -1 : Kept.
         0 : Empty or suppressed.
         1 : To be processed (converted to either kept or supressed).

        NOTE: The NMS first rounds points to integers, so NMS distance might not
        be exactly dist_thresh. It also assumes points are within image boundaries.

        Inputs
          in_corners - 3xN numpy array with corners [x_i, y_i, confidence_i]^T.
          H - Image height.
          W - Image width.
          dist_thresh - Distance to suppress, measured as an infinty norm distance.
        Returns
          nmsed_corners - 3xN numpy matrix with surviving corners.
          nmsed_inds - N length numpy vector with surviving corner indices.
        """
        grid = np.zeros((H, W)).astype(int)  # Track NMS data.
        inds = np.zeros((H, W)).astype(int)  # Store indices of points.
        # Sort by confidence and round to nearest int.
        inds1 = np.argsort(-in_corners[2, :])
        corners = in_corners[:, inds1]
        rcorners = corners[:2, :].round().astype(int)  # Rounded corners.
        # Check for edge case of 0 or 1 corners.
        if rcorners.shape[1] == 0:
            return np.zeros((3, 0)).astype(int), np.zeros(0).astype(int)
        if rcorners.shape[1] == 1:
            out = np.vstack((rcorners, in_corners[2])).reshape(3, 1)
            return out, np.zeros(1).astype(int)
        # Initialize the grid.
        for i, rc in enumerate(rcorners.T):
            grid[rcorners[1, i], rcorners[0, i]] = 1
            inds[rcorners[1, i], rcorners[0, i]] = i
        # Pad the border of the grid, so that we can NMS points near the border.
        pad = dist_thresh
        grid = np.pad(grid, ((pad, pad), (pad, pad)), mode="constant")
        # Iterate through points, highest to lowest conf, suppress neighborhood.
        count = 0
        for i, rc in enumerate(rcorners.T):
            # Account for top and left padding.
            pt = (rc[0] + pad, rc[1] + pad)
            if grid[pt[1], pt[0]] == 1:  # If not yet suppressed.
                grid[pt[1] - pad : pt[1] + pad + 1, pt[0] - pad : pt[0] + pad + 1] = 0
                grid[pt[1], pt[0]] = -1
                count += 1
        # Get all surviving -1's and return sorted array of remaining corners.
        keepy, keepx = np.where(grid == -1)
        keepy, keepx = keepy - pad, keepx - pad
        inds_keep = inds[keepy, keepx]
        out = corners[:, inds_keep]
        values = out[-1, :]
        inds2 = np.argsort(-values)
        out = out[:, inds2]
        out_inds = inds1[inds_keep[inds2]]
        return out, out_inds
