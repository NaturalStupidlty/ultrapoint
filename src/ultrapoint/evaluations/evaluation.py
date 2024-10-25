import os
import argparse
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from loguru import logger
from numpy.linalg import inv

from ultrapoint.evaluations.descriptor_evaluation import compute_homography
from ultrapoint.evaluations.detector_evaluation import compute_repeatability
from ultrapoint.utils.draw import plot_imgs, draw_keypoints
from ultrapoint.utils.utils import warp_points
from ultrapoint.utils.utils import filter_points


def to3dim(img):
    if img.ndim == 2:
        img = img[:, :, np.newaxis]
    return img


def draw_matches_cv(data, matches, plot_points=True):
    if plot_points:
        keypoints1 = [cv2.KeyPoint(p[1], p[0], 1) for p in data["keypoints1"]]
        keypoints2 = [cv2.KeyPoint(p[1], p[0], 1) for p in data["keypoints2"]]
    else:
        matches_pts = data["matches"]
        keypoints1 = [cv2.KeyPoint(p[0], p[1], 1) for p in matches_pts]
        keypoints2 = [cv2.KeyPoint(p[2], p[3], 1) for p in matches_pts]
        print(f"matches_pts: {matches_pts}")

    img1 = to3dim(data["image1"])
    img2 = to3dim(data["image2"])
    img1 = np.concatenate([img1, img1, img1], axis=2).astype(np.uint8)
    img2 = np.concatenate([img2, img2, img2], axis=2).astype(np.uint8)
    return cv2.drawMatches(
        img1,
        keypoints1,
        img2,
        keypoints2,
        matches,
        None,
        matchColor=(0, 255, 0),
        singlePointColor=(0, 0, 255),
    )


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


def plot_sample(data, repeatability, save_path: str = None):
    pts = data["prob"]
    img1 = draw_keypoints(
        data["image"].squeeze(),
        pts,
        np.ones(pts.shape[0]),
    )

    pts = data["warped_prob"]
    img2 = draw_keypoints(data["warped_image"].squeeze(), pts, np.ones(pts.shape[0]))

    plot_imgs(
        [img1.astype(np.uint8), img2.astype(np.uint8)],
        titles=["img1", "img2"],
        dpi=200,
    )
    plt.title(f"Repeatability: {repeatability}")
    plt.tight_layout()

    plt.savefig(save_path, dpi=300, bbox_inches="tight")


def evaluate(
    descriptors_path: str,
    repeatability_threshold: int = 3,
    inliers_method: str = "cv",
    verbose: bool = True,
):
    correctness = []
    localization_error = []
    repeatability = []
    matching_score = []
    mAP = []

    if args.outputImg:
        path_warp = os.path.join(descriptors_path, "warping")
        path_match = os.path.join(descriptors_path, "matching")
        path_rep = os.path.join(descriptors_path, "repeatibility")
        os.makedirs(path_warp, exist_ok=True)
        os.makedirs(path_match, exist_ok=True)
        os.makedirs(path_rep, exist_ok=True)

    logger.info(f"Path: {descriptors_path}")
    files = find_files_with_ext(descriptors_path)[:3]
    files.sort(key=lambda x: int(x.stem))

    for file in tqdm(files):
        data = np.load(file)
        logger.debug(f"Loaded {file} successfully.")

        real_H = data["homography"]
        image = data["image"]
        warped_image = data["warped_image"]
        keypoints = data["prob"]
        warped_keypoints = data["warped_prob"]

        if args.repeatibility:
            rep, local_error = compute_repeatability(
                data,
                distance_thresh=repeatability_threshold,
            )
            repeatability.append(rep)
            if local_error > 0:
                localization_error.append(local_error)

            logger.debug(f"Repeatability: {rep}")
            logger.debug(f"Localization error: {local_error}")

            if args.outputImg:
                plot_sample(
                    data, repeatability, os.path.join(path_rep, file.stem + ".png")
                )

        if args.homography:
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

            print("m. score: ", score)
            matching_score.append(score)

            def getMatches(data):
                from src.ultrapoint.models import PointTracker

                descriptor = data["descriptor"]
                warped_descriptor = data["warped_descriptor"]

                nn_thresh = 1.2
                print("nn threshold: ", nn_thresh)
                tracker = PointTracker(max_length=2, nn_thresh=nn_thresh)
                tracker.update(keypoints.T, descriptor.T)
                tracker.update(warped_keypoints.T, warped_descriptor.T)
                matches = tracker.get_matches().T
                mscores = tracker.get_mscores().T

                # mAP
                print("matches: ", matches.shape)
                print("mscores: ", mscores.shape)
                print("mscore max: ", mscores.max(axis=0))
                print("mscore min: ", mscores.min(axis=0))

                return matches, mscores

            def getInliers(matches, H, epi=3, verbose=False):
                """
                input:
                    matches: numpy (n, 4(x1, y1, x2, y2))
                    H (ground truth homography): numpy (3, 3)
                """
                from src.ultrapoint.evaluations import warp_keypoints

                # warp points
                warped_points = warp_keypoints(
                    matches[:, :2], H
                )  # make sure the input fits the (x,y)

                # compute point distance
                norm = np.linalg.norm(warped_points - matches[:, 2:4], ord=None, axis=1)
                inliers = norm < epi
                if verbose:
                    print(
                        "Total matches: ",
                        inliers.shape[0],
                        ", inliers: ",
                        inliers.sum(),
                        ", percentage: ",
                        inliers.sum() / inliers.shape[0],
                    )

                return inliers

            def getInliers_cv(matches, H=None, epi=3, verbose=False):
                import cv2

                # count inliers: use opencv homography estimation
                # Estimate the homography between the matches using RANSAC
                H, inliers = cv2.findHomography(
                    matches[:, [0, 1]], matches[:, [2, 3]], cv2.RANSAC
                )
                inliers = inliers.flatten()
                print(
                    "Total matches: ",
                    inliers.shape[0],
                    ", inliers: ",
                    inliers.sum(),
                    ", percentage: ",
                    inliers.sum() / inliers.shape[0],
                )
                return inliers

            def computeAP(m_test, m_score):
                from sklearn.metrics import average_precision_score

                average_precision = average_precision_score(m_test, m_score)
                print(
                    "Average precision-recall score: {0:0.2f}".format(average_precision)
                )
                return average_precision

            def flipArr(arr):
                return arr.max() - arr

            if args.sift:
                assert result is not None
                matches, mscores = result["matches"], result["mscores"]
            else:
                matches, mscores = getMatches(data)

            real_H = data["homography"]
            if inliers_method == "gt":
                # use ground truth homography
                print("use ground truth homography for inliers")
                inliers = getInliers(matches, real_H, epi=3, verbose=verbose)
            else:
                # use opencv estimation as inliers
                print("use opencv estimation for inliers")
                inliers = getInliers_cv(matches, real_H, epi=3, verbose=verbose)

            ## distance to confidence
            if args.sift:
                m_flip = flipArr(mscores[:])  # for sift
            else:
                m_flip = flipArr(mscores[:, 2])

            ap = 0
            if inliers.shape[0] > 0 and inliers.sum() > 0:
                ap = computeAP(inliers, m_flip)

            mAP.append(ap)

            if args.outputImg:
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

                ## plot filtered image
                img1, img2 = data["image"].squeeze(), data["warped_image"].squeeze()
                warped_img1 = cv2.warpPerspective(
                    img1, H, (img2.shape[1], img2.shape[0])
                )
                plot_imgs(
                    [img1, img2, warped_img1],
                    titles=["img1", "img2", "warped_img1"],
                    dpi=200,
                )
                plt.tight_layout()
                plt.savefig(path_warp + "/" + file.stem + ".png")

                # draw matches
                result["image1"] = image.squeeze()
                result["image2"] = warped_image.squeeze()
                matches = np.array(result["cv2_matches"])
                ratio = 0.2
                ran_idx = np.random.choice(
                    matches.shape[0], int(matches.shape[0] * ratio)
                )

                img = draw_matches_cv(result, matches[ran_idx], plot_points=True)
                # filename = "correspondence_visualization"
                plot_imgs([img], titles=["Two images feature correspondences"], dpi=200)
                plt.tight_layout()
                plt.savefig(
                    path_match + "/" + file.stem + "cv.png", bbox_inches="tight"
                )
                plt.close("all")

        if args.plotMatching:
            matches = result["matches"]  # np [N x 4]
            if matches.shape[0] > 0:
                from src.ultrapoint.utils.draw import draw_matches

                filename = path_match + "/" + file.stem + "m.png"
                ratio = 0.1
                inliers = result["inliers"]

                matches_in = matches[inliers == True]
                matches_out = matches[inliers == False]

                def get_random_m(matches, ratio):
                    ran_idx = np.random.choice(
                        matches.shape[0], int(matches.shape[0] * ratio)
                    )
                    return matches[ran_idx], ran_idx

                image = data["image"].squeeze()
                warped_image = data["warped_image"].squeeze()

                ## outliers
                matches_temp, _ = get_random_m(matches_out, ratio)
                draw_matches(
                    image,
                    warped_image,
                    matches_temp,
                    lw=0.5,
                    color="r",
                    filename=None,
                    show=False,
                    if_fig=True,
                )

                ## inliers
                matches_temp, _ = get_random_m(matches_in, ratio)
                draw_matches(
                    image,
                    warped_image,
                    matches_temp,
                    lw=1.0,
                    filename=filename,
                    show=False,
                    if_fig=False,
                )

    if args.repeatibility:
        logger.info(f"Repeatability threshold: {repeatability_threshold}")
        logger.info(f"Mean repeatability: {np.array(repeatability).mean()}")
        logger.info(f"Mean localization error: {np.array(localization_error).mean()}")

    if args.homography:
        logger.info("Homography estimation:")
        logger.info(f"Homography threshold: {homography_thresh}")
        logger.info(f"Mean correctness: {np.array(correctness).mean(axis=0)}")
        logger.info(f"mAP: {np.array(mAP).mean()}")
        logger.info(f"Matching score: {np.array(matching_score).mean(axis=0)}")

    for i, file in enumerate(files):
        logger.debug(file)
        if args.repeatibility:
            logger.debug(f"Repeatability: {repeatability[i]}")
        if args.homography:
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
    parser.add_argument("-o", "--outputImg", action="store_true")
    parser.add_argument("-r", "--repeatibility", action="store_true")
    parser.add_argument("-homo", "--homography", action="store_true")
    parser.add_argument("-plm", "--plotMatching", action="store_true")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    evaluate(args.descriptors_path)
