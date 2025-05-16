import os
import argparse
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from numpy.linalg import inv

from ultrapoint.loggers.loguru import logger, create_logger
from ultrapoint.evaluations.descriptor_evaluation import compute_homography
from ultrapoint.evaluations.detector_evaluation import (
    compute_repeatability,
    warp_keypoints,
)
from ultrapoint.utils.draw import plot_imgs, draw_keypoints
from ultrapoint.utils.utils import warp_points
from ultrapoint.utils.utils import filter_points


def to3dim(img):
    if img.ndim == 2:
        img = img[:, :, np.newaxis]
    return img


def find_files_with_ext(directory: str, extension: str = ".npz"):
    return [
        file
        for file in Path(directory).iterdir()
        if file.is_file() and file.suffix == extension
    ]


def warpLabels(pnts, homography, H, W):
    import torch

    """
    input:
        pnts: numpy
        homography: numpy
    output:
        warped_pnts: numpy
    """
    pnts = torch.tensor(pnts).long()
    homography = torch.tensor(homography, dtype=torch.float32)
    warped_pnts = warp_points(
        torch.stack((pnts[:, 0], pnts[:, 1]), dim=1), homography
    )  # check the (x, y)
    warped_pnts = filter_points(warped_pnts, torch.tensor([W, H])).round().long()
    return warped_pnts.numpy()


def evaluate(
    descriptors_path: str,
    repeatability_threshold: int = 3,
    inliers_method: str = "cv",
    verbose: bool = True,
    do_repeatability: bool = False,
    do_homography: bool = False,
    do_plot_matching: bool = False,
    output_img: bool = False,
    use_sift: bool = False,
):
    """
    Run the chosen evaluation sub-tasks and save metrics/visualisations.

    Parameters
    ----------
    descriptors_path : str
        Folder containing <index>.npz files.
    repeatability_threshold : int
        Pixel distance for considering two keypoints the “same”.
    inliers_method : {"cv", "gt"}
        How to decide which matches are inliers.
    verbose : bool
        Toggle loguru DEBUG output.
    do_repeatability, do_homography, do_plot_matching, output_img, use_sift : bool
        Feature switches corresponding to the CLI flags.
    """
    if verbose:
        logger.enable(__name__)
    else:
        logger.disable(__name__)

    correctness = []
    localization_error = []
    repeatability = []
    matching_score = []
    mAP = []

    if output_img:
        path_warp = os.path.join(descriptors_path, "warping")
        path_match = os.path.join(descriptors_path, "matching")
        os.makedirs(path_warp, exist_ok=True)
        os.makedirs(path_match, exist_ok=True)

    logger.info(f"Path: {descriptors_path}")
    files = find_files_with_ext(descriptors_path)
    files = [file for file in files if file.stem.isnumeric()]
    files.sort(key=lambda x: int(x.stem))

    assert len(files) > 0, f"No files found in {descriptors_path} with extension .npz"

    for file in tqdm(files):
        data = np.load(file)
        logger.debug(f"Loaded {file} successfully.")

        real_H = data["homography"]
        image = data["image"]
        warped_image = data["warped_image"]
        keypoints = data["keypoints"]
        warped_keypoints = data["warped_keypoints"]

        if do_repeatability:
            rep, local_error = compute_repeatability(
                data,
                distance_thresh=repeatability_threshold,
            )
            repeatability.append(rep)
            if local_error > 0:
                localization_error.append(local_error)

            logger.debug(f"Repeatability: {rep}")
            logger.debug(f"Localization error: {local_error}")

        if do_homography:
            homography_thresh = [1, 3, 5, 10, 20, 50]
            result = compute_homography(data, correctness_thresh=homography_thresh)
            correctness.append(result["correctness"])

            H, W = image.shape
            filtered_unwarped_keypoints = warpLabels(
                warped_keypoints, inv(real_H), H, W
            )

            score = (result["inliers"].sum() * 2) / (
                keypoints.shape[0] + filtered_unwarped_keypoints.shape[0]
            )

            logger.debug(f"Matching score: {score}")
            matching_score.append(score)

            def getMatches(data):
                from ultrapoint.evaluations import PointTracker

                descriptor = data["descriptor"]
                warped_descriptor = data["warped_descriptor"]

                nn_thresh = 1.2
                logger.debug(f"nn threshold: {nn_thresh}")
                tracker = PointTracker(max_length=2, nn_thresh=nn_thresh)
                tracker.update(keypoints, descriptor)
                tracker.update(warped_keypoints, warped_descriptor)
                matches = tracker.get_matches().T
                mscores = tracker.get_mscores().T

                # mAP
                logger.debug(f"matches shape: {matches.shape}")
                logger.debug(f"mscores shape: {mscores.shape}")
                logger.debug(f"mscore max: {mscores.max(axis=0)}")
                logger.debug(f"mscore min: {mscores.min(axis=0)}")

                return matches, mscores

            def getInliers(matches, H, epi=3):
                """
                input:
                    matches: numpy (n, 4(x1, y1, x2, y2))
                    H (ground truth homography): numpy (3, 3)
                """
                # warp points
                warped_points = warp_keypoints(
                    matches[:, :2], H
                )  # make sure the input fits the (x,y)

                # compute point distance
                norm = np.linalg.norm(warped_points - matches[:, 2:4], ord=None, axis=1)
                inliers = norm < epi

                logger.debug(
                    f"Total matches: {inliers.shape[0]}, inliers: {inliers.sum()}, percentage: {inliers.sum() / inliers.shape[0]}"
                )

                return inliers

            def getInliers_cv(matches):
                # count inliers: use opencv homography estimation
                # Estimate the homography between the matches using RANSAC
                H, inliers = cv2.findHomography(
                    matches[:, [0, 1]], matches[:, [2, 3]], cv2.RANSAC
                )
                inliers = inliers.flatten()

                logger.debug(
                    f"Total matches: {inliers.shape[0]}, inliers: {inliers.sum()}, percentage: {inliers.sum() / inliers.shape[0]}"
                )

                return inliers

            def computeAP(m_test, m_score):
                from sklearn.metrics import average_precision_score

                average_precision = average_precision_score(m_test, m_score)
                logger.debug(
                    f"mAP score: {average_precision:0.2f}"
                )
                return average_precision

            def flipArr(arr):
                return arr.max() - arr

            if use_sift:
                assert result is not None
                matches, mscores = result["matches"], result["mscores"]
            else:
                matches, mscores = getMatches(data)

            if inliers_method == "gt":
                # use ground truth homography
                logger.debug("Using ground truth homography for inliers")
                inliers = getInliers(matches, real_H, epi=3)
            else:
                # use opencv estimation as inliers
                logger.debug("Using OpenCV estimation for inliers")
                inliers = getInliers_cv(matches)

            # distance to confidence
            if use_sift:
                m_flip = flipArr(mscores[:])  # for sift
            else:
                m_flip = flipArr(mscores[:, 2])

            ap = 0
            if inliers.shape[0] > 0 and inliers.sum() > 0:
                ap = computeAP(inliers, m_flip)

            mAP.append(ap)

            if output_img:
                # draw warping
                output = result
                img1 = image.squeeze()
                img2 = warped_image.squeeze()

                img1 = to3dim(img1)
                img2 = to3dim(img2)
                H = output["homography"]
                warped_img1 = cv2.warpPerspective(
                    img1, H, (img2.shape[1], img2.shape[0])
                )

                img1 = np.concatenate([img1, img1, img1], axis=2)
                warped_img1 = np.stack([warped_img1, warped_img1, warped_img1], axis=2)
                img2 = np.concatenate([img2, img2, img2], axis=2)
                plot_imgs(
                    [img1, img2, warped_img1],
                    titles=["img1", "img2", "warped_img1"],
                    dpi=200,
                )
                plt.tight_layout()
                plt.savefig(path_warp + "/" + file.stem + ".png")
                plt.close()

        if do_plot_matching:
            matches = result["matches"]  # np [N x 4]
            if matches.shape[0] > 0:
                from src.ultrapoint.utils.draw import draw_matches
                image = data["image"].squeeze()
                warped_image = data["warped_image"].squeeze()
                draw_matches(
                    image,
                    warped_image,
                    matches,
                    kp1=keypoints,
                    kp2=warped_keypoints,
                    lw=1.0,
                    color="g",
                    unmatched_color_left="r",
                    unmatched_color_right="r",
                    filename=path_match + "/" + file.stem + "m.png",
                )

    if repeatability:
        logger.info(f"Repeatability threshold: {repeatability_threshold}")
        logger.info(f"Mean repeatability: {np.array(repeatability).mean()}")
        logger.info(f"Mean localization error: {np.array(localization_error).mean()}")

    if do_homography:
        logger.info("Homography estimation:")
        logger.info(f"Homography threshold: {homography_thresh}")
        logger.info(f"Mean correctness: {np.array(correctness).mean(axis=0)}")
        logger.info(f"mAP: {np.array(mAP).mean()}")
        logger.info(f"Matching score: {np.array(matching_score).mean(axis=0)}")

    for i, file in enumerate(files):
        logger.debug(file)
        if repeatability:
            logger.debug(f"Repeatability: {repeatability[i]}")
        if do_homography:
            logger.debug(f"Correctness: {correctness[i]}")
            logger.debug(f"Matching score: {matching_score[i]}")
            logger.debug(f"mAP: {mAP[i]}")

    dict_of_lists = {
        "repeatability": repeatability,
        "localization_err": localization_error,
        "correctness": np.array(correctness),
        "homography_thresh": homography_thresh,
        "mscore": matching_score,
        "mAP": np.array(mAP),
    }

    results_path = os.path.join(descriptors_path, "evaluation.npz")
    logger.info(f"Saving evaluation results to {results_path}")
    np.savez(
        results_path,
        **dict_of_lists,
    )


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("descriptors_path", type=str)
    parser.add_argument("--sift", action="store_true", help="use sift matches")
    parser.add_argument(
        "--repeatability_threshold",
        type=int,
        default=3,
        help="Pixel distance for considering two keypoints the same.",
    )
    parser.add_argument(
        "--inliers_method",
        type=str,
        default="cv",
        choices=["cv", "gt"],
        help="How to decide which matches are inliers.",
    )
    parser.add_argument("-o", "--output_img", action="store_true")
    parser.add_argument("-r", "--repeatability", action="store_true")
    parser.add_argument("-homo", "--homography", action="store_true")
    parser.add_argument("-plm", "--plot_matching", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    create_logger(level="INFO", directory="../assets/logs/evaluation")
    evaluate(
        descriptors_path=str(args.descriptors_path),
        repeatability_threshold=args.repeatability_threshold,
        inliers_method=args.inliers_method,
        verbose=args.verbose,
        do_repeatability=args.repeatability,
        do_homography=args.homography,
        do_plot_matching=args.plot_matching,
        output_img=args.output_img,
        use_sift=args.sift,
    )
