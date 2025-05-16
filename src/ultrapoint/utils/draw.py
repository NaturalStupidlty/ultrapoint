"""util functions for visualization

"""

import numpy as np
import cv2
import matplotlib.pyplot as plt


def plot_imgs(
    imgs, titles=None, cmap="brg", ylabel="", normalize=False, ax=None, dpi=100
):
    n = len(imgs)
    if not isinstance(cmap, list):
        cmap = [cmap] * n
    if ax is None:
        fig, ax = plt.subplots(1, n, figsize=(6 * n, 6), dpi=dpi)
        if n == 1:
            ax = [ax]
    else:
        if not isinstance(ax, list):
            ax = [ax]
        assert len(ax) == len(imgs)
    for i in range(n):
        if imgs[i].shape[-1] == 3:
            imgs[i] = imgs[i][..., ::-1]  # BGR to RGB
        ax[i].imshow(
            imgs[i],
            cmap=plt.get_cmap(cmap[i]),
            vmin=None if normalize else 0,
            vmax=None if normalize else 1,
        )
        if titles:
            ax[i].set_title(titles[i])
        ax[i].get_yaxis().set_ticks([])
        ax[i].get_xaxis().set_ticks([])
        for spine in ax[i].spines.values():  # remove frame
            spine.set_visible(False)
    ax[0].set_ylabel(ylabel)
    plt.tight_layout()


# from utils.draw import img_overlap
def img_overlap(img_r, img_g, img_gray):  # img_b repeat
    img = np.concatenate((img_gray, img_gray, img_gray), axis=0)
    img[0, :, :] += img_r[0, :, :]
    img[1, :, :] += img_g[0, :, :]
    img[img > 1] = 1
    img[img < 0] = 0
    return img


def draw_keypoints(img, points, scores, color=(0, 255, 0), radius=3, resize=3):
    """
    Draw keypoints on an image with transparency depending on the score.

    :param img: Input image (grayscale or RGB) (numpy [H, W] or [H, W, 3])
    :param points: Points with scores (numpy [N, 2] where each point is (x, y))
    :param scores: Scores of the keypoints (numpy [N])
    :param color: Color of the keypoints (default: green)
    :param radius: Radius of the keypoints (default: 3)
    :param resize: Resize factor for the image and points (default: 1)
    :return: Image with keypoints drawn
    """
    if len(img.shape) == 2 or img.shape[2] == 1:
        img = np.repeat(
            cv2.resize(img, None, fx=resize, fy=resize)[..., np.newaxis], 3, -1
        )
    else:
        img = cv2.resize(img, None, fx=resize, fy=resize)

    if len(points) == 0:
        return img

    overlay = img.copy()
    for (x, y), score in zip(points, scores):
        alpha = np.clip(score, 0, 1)  # Ensure the score is in [0, 1] range
        point_color = (
            int(color[0]),
            int(color[1]),
            int(color[2]),
        )

        # Draw the keypoint on the overlay with full opacity
        cv2.circle(
            overlay,
            (int(x * resize), int(y * resize)),
            radius,
            point_color,
            thickness=-1,
        )

        # Blend the overlay and the original image using alpha
        img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

    return img


def draw_matches(
    rgb1,
    rgb2,
    match_pairs,
    *,
    kp1=None,                     # ← all keypoints in rgb1  (shape M₁×2, [x, y])
    kp2=None,                     # ← all keypoints in rgb2  (shape M₂×2, [x, y])
    lw=0.5,
    color=None,                   # if None ⇒ a different random colour per match
    unmatched_color_left="b",     # colour for LEFT-image unmatched keypoints
    unmatched_color_right="y",    # colour for RIGHT-image unmatched keypoints
    unmatched_marker_left="o",
    unmatched_marker_right="o",
    unmatched_size=10,
    if_fig=True,
    filename="matches.png",
    show=False,
    seed=None,                    # optional RNG seed for reproducibility
):
    """Visualise correspondences *and* (optionally) unmatched key‑points.

    Compared with the original implementation, **each correspondence line is now
    drawn with its *own* random colour** (unless *color* is set to a specific
    colour).  The random generator can be fixed via *seed* for deterministic
    results.

    Parameters
    ----------
    rgb1, rgb2 : ndarray
        Left and right images.  Grayscale (H×W) or RGB (H×W×3).
    match_pairs : (N, 4) ndarray
        Rows are (x₁, y₁, x₂, y₂) for each match.
    kp1, kp2 : (M₁, 2) / (M₂, 2) ndarray, optional
        Full keypoint lists for rgb1 / rgb2.  If given, keypoints absent
        from *match_pairs* are rendered with *unmatched_color_left/right*.
        (If either array is omitted, unmatched points for that image are not
        shown.)
    color : Any valid Matplotlib colour spec or *None*
        If a specific colour is given, **all** match lines use it.  If *None*
        (default) the colour is chosen independently at random for every match.

    All other arguments keep their previous meaning.
    """

    # ---------------------------------------------------------------------
    # deterministic randomness (optional)
    if seed is not None:
        np.random.seed(seed)

    # ---- build side‑by‑side canvas --------------------------------------
    rgb1_disp = rgb1 if rgb1.ndim == 3 else np.repeat(rgb1[..., None], 3, axis=-1)
    rgb2_disp = rgb2 if rgb2.ndim == 3 else np.repeat(rgb2[..., None], 3, axis=-1)

    h1, w1 = rgb1_disp.shape[:2]
    h2, w2 = rgb2_disp.shape[:2]

    canvas = np.zeros((max(h1, h2), w1 + w2, 3), dtype=rgb1_disp.dtype)
    canvas[:h1, :w1] = rgb1_disp
    canvas[:h2, w1:] = rgb2_disp

    if if_fig:
        plt.figure(figsize=(15, 5))
    plt.axis("off")
    plt.imshow(canvas, zorder=1)

    # ---- draw the matches (each line gets its own colour) ---------------
    for x1, y1, x2, y2 in match_pairs:
        this_col = np.random.rand(3) if color is None else color
        plt.plot(
            [x1, x2 + w1],
            [y1, y2],
            linestyle="-",
            linewidth=lw,
            aa=False,
            marker="o",
            markersize=2,
            fillstyle="none",
            color=this_col,
            zorder=2,
        )

    # ---- helper: find unmatched -----------------------------------------
    def _unmatched(all_kp, matched_kp):
        """Return keypoints in *all_kp* that do **not** occur in *matched_kp*."""
        if all_kp is None or all_kp.size == 0:
            return np.empty((0, 2))
        # round to 3 dp → stable hashing even for float coords
        key_set = {tuple(p) for p in np.round(all_kp, 3)}
        matched_set = {tuple(p) for p in np.round(matched_kp, 3)}
        return np.array([p for p in key_set.difference(matched_set)])

    # ---- draw unmatched keypoints (if kp1 / kp2 supplied) ---------------
    unmatched_left = _unmatched(kp1, match_pairs[:, 0:2])
    if unmatched_left.size:
        plt.scatter(
            unmatched_left[:, 0],
            unmatched_left[:, 1],
            s=unmatched_size,
            marker=unmatched_marker_left,
            c=unmatched_color_left,
            linewidths=0,
            zorder=3,
        )

    unmatched_right = _unmatched(kp2, match_pairs[:, 2:4])
    if unmatched_right.size:
        plt.scatter(
            unmatched_right[:, 0] + w1,      # shift to canvas coords
            unmatched_right[:, 1],
            s=unmatched_size,
            marker=unmatched_marker_right,
            c=unmatched_color_right,
            linewidths=0,
            zorder=3,
        )

    # ---- finish up -------------------------------------------------------
    plt.tight_layout()
    if filename is not None:
        plt.savefig(filename, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()
