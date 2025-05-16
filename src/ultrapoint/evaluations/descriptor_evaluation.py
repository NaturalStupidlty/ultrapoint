import cv2
import numpy as np
from loguru import logger


def compute_homography(data, correctness_thresh=3, shape=(240, 320)):
    """
    Compute the homography between 2 sets of detections and descriptors inside data.
    """
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    cv2_matches = bf.match(data["descriptor"], data["warped_descriptor"])

    # Keeps only the points shared between the two views
    m_dist = np.array([m.distance for m in cv2_matches])
    query_matches = np.array([m.queryIdx for m in cv2_matches])
    train_matches = np.array([m.trainIdx for m in cv2_matches])
    m_keypoints = data["keypoints"][query_matches, :]
    m_warped_keypoints = data["warped_keypoints"][train_matches, :]

    matches = np.hstack((m_keypoints, m_warped_keypoints))
    logger.debug(f"Homography matches: {matches.shape}")

    # Estimate the homography between the matches using RANSAC
    H, inliers = cv2.findHomography(m_keypoints, m_warped_keypoints, cv2.RANSAC)
    inliers = inliers.flatten()

    # Compute correctness
    if H is None:
        mean_dist = 0
        correctness = 0
        H = np.identity(3)
        logger.debug("No valid estimation")
    else:
        corners = np.array(
            [
                [0, 0, 1],
                [0, shape[0] - 1, 1],
                [shape[1] - 1, 0, 1],
                [shape[1] - 1, shape[0] - 1, 1],
            ]
        )
        real_warped_corners = np.dot(
            corners, np.transpose(data["homography"])
        ).squeeze()
        real_warped_corners = real_warped_corners[:, :2] / real_warped_corners[:, 2:]
        logger.debug(f"Real warped corners: {real_warped_corners}")

        warped_corners = np.dot(corners, np.transpose(H))
        warped_corners = warped_corners[:, :2] / warped_corners[:, 2:]
        logger.debug(f"Warped corners: {warped_corners}")

        mean_dist = np.mean(
            np.linalg.norm(real_warped_corners - warped_corners, axis=1)
        )
        correctness = mean_dist <= correctness_thresh

    return {
        "correctness": correctness,
        "keypoints1": data["keypoints"],
        "keypoints2": data["warped_keypoints"],
        "matches": matches,  # cv2.match
        "cv2_matches": cv2_matches,
        "mscores": m_dist / (m_dist.max()),  # normalized distance
        "inliers": inliers,
        "homography": H,
        "mean_dist": mean_dist,
    }
