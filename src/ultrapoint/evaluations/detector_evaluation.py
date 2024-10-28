import os
from glob import glob

import numpy as np


def warp_keypoints(keypoints, H):
    """
    :param keypoints:
    points:
        numpy (N, (x,y))
    :param H:
    :return:
    """
    num_points = keypoints.shape[0]
    homogeneous_points = np.concatenate([keypoints, np.ones((num_points, 1))], axis=1)
    warped_points = np.dot(homogeneous_points, np.transpose(H)).squeeze()
    return warped_points[:, :2] / warped_points[:, 2:]


def compute_repeatability(data, distance_thresh=3):
    """
    Compute the repeatability. The experiment must contain in its output the prediction
    on 2 images, an original image and a warped version of it, plus the homography
    linking the 2 images.
    """

    def filter_keypoints(points, shape):
        """Keep only the points whose coordinates are
        inside the dimensions of shape."""
        """
        points:
            numpy (N, (x,y))
        shape:
            (y, x)
        """
        mask = (
            (points[:, 0] >= 0)
            & (points[:, 0] < shape[1])
            & (points[:, 1] >= 0)
            & (points[:, 1] < shape[0])
        )
        return points[mask, :]

    def keep_true_keypoints(points, H, shape):
        """Keep only the points whose warped coordinates by H
        are still inside shape."""
        """
        input:
            points: numpy (N, (x,y))
            shape: (y, x)
        return:
            points: numpy (N, (x,y))
        """
        warped_points = warp_keypoints(points[:, :2], H).squeeze()
        mask = (
            (warped_points[:, 0] >= 0)
            & (warped_points[:, 0] < shape[1])
            & (warped_points[:, 1] >= 0)
            & (warped_points[:, 1] < shape[0])
        )
        return points[mask, :]

    warped_keypoints = keep_true_keypoints(
        data["warped_prob"], np.linalg.inv(data["homography"]), data["image"].shape
    )
    true_warped_keypoints = warp_keypoints(data["prob"], data["homography"])
    true_warped_keypoints = filter_keypoints(true_warped_keypoints, data["image"].shape)
    norms = np.linalg.norm(
        true_warped_keypoints[:, np.newaxis, :] - warped_keypoints[np.newaxis, :, :],
        axis=2,
    )

    repeatability = 0
    localization_err = 0
    if (
        norms.shape[1] > 0
        and true_warped_keypoints.shape[0] + warped_keypoints.shape[0] > 0
    ):
        min1 = np.min(norms, axis=1)
        count1 = np.sum(min1 <= distance_thresh)
        local_error1 = min1[min1 <= distance_thresh].sum()

        min2 = np.min(norms, axis=0)
        count2 = np.sum(min2 <= distance_thresh)
        local_error2 = min2[min2 <= distance_thresh].sum()

        repeatability = (count1 + count2) / (
            true_warped_keypoints.shape[0] + warped_keypoints.shape[0]
        )

        localization_err += local_error1 / (count1 + count2)
        localization_err += local_error2 / (count1 + count2)

    return repeatability, localization_err
