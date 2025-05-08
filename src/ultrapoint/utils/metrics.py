import numpy


def precision_recall_vectors(predictions_keypoints, predictions_scores, labels_keypoints, dist_thresh=3):
    """
    Compute precision & recall given:
      - predictions_keypoints: (N,2) array of [x,y] predicted keypoints
      - predictions_scores: (N,) confidences
      - labels_keypoints: (M,2) array of [x,y] ground-truth keypoints
      - dist_thresh: maximum pixel distance to count a match

    Returns:
      precision: array of length N+2
      recall:    array of length N+2
    """
    # ensure numpy arrays
    predictions_keypoints = numpy.asarray(predictions_keypoints)
    predictions_scores = numpy.asarray(predictions_scores)
    labels_keypoints = numpy.asarray(labels_keypoints)

    # sort predictions by descending confidence
    order = numpy.argsort(predictions_scores)[::-1]
    predictions_keypoints = predictions_keypoints[order]
    predictions_scores = predictions_scores[order]

    N = len(predictions_keypoints)
    M = len(labels_keypoints)

    # ---------- special case: no ground truth ----------
    if M == 0:
        if N == 0:
            # perfect silence: precision=1, recall=0 everywhere
            return numpy.array([1., 0.]), numpy.array([0., 0.])
        else:
            # every prediction is a false positive
            fp_cum = numpy.arange(1, N + 1)
            tp_cum = numpy.zeros_like(fp_cum)

            precision_raw = tp_cum / (fp_cum + 1e-8)
            recall_raw = numpy.zeros_like(precision_raw)

            precision = numpy.concatenate(([1.], precision_raw, [0.]))
            recall = numpy.concatenate(([0.], recall_raw,    [0.]))

            return precision, recall

    tp = numpy.zeros(N, dtype=int)
    fp = numpy.zeros(N, dtype=int)
    matched = numpy.zeros(M, dtype=bool)

    for i, (x, y) in enumerate(predictions_keypoints):
        if M == 0:
            # no ground truth → everything is false positive
            fp[i] = 1
            continue

        # compute all distances to GT points
        dists = numpy.hypot(labels_keypoints[:, 0] - x, labels_keypoints[:, 1] - y)
        j = numpy.argmin(dists)
        if dists[j] <= dist_thresh and not matched[j]:
            tp[i] = 1
            matched[j] = True
        else:
            fp[i] = 1

    # cumulative sums
    tp_cum = numpy.cumsum(tp)
    fp_cum = numpy.cumsum(fp)

    # recall and precision curves
    recall = tp_cum / (M if M > 0 else 1)
    precision = tp_cum / (tp_cum + fp_cum + 1e-8)

    # add endpoints and do the usual monotonicity fix
    recall = numpy.concatenate(([0.], recall, [1.]))
    precision = numpy.concatenate(([1.], precision, [0.]))
    precision = numpy.maximum.accumulate(precision[::-1])[::-1]

    return precision, recall


def average_precision(precision, recall):
    """
    Compute the Average Precision (AP) as the area under the precision–recall curve.
    Assumes precision and recall arrays include the [0,1] endpoints and that
    precision is monotonically decreasing (you’ve already applied the
    `numpy.maximum.accumulate(...)[::-1]` trick).

    AP = Σᵢ (Rᵢ − Rᵢ₋₁) * Pᵢ
    """
    # ensure numpy arrays
    precision = numpy.asarray(precision)
    recall = numpy.asarray(recall)

    # differences in recall
    d_recall = recall[1:] - recall[:-1]

    # sum up area under curve
    return numpy.sum(d_recall * precision[1:])


def compute_metrics(predictions_keypoints, predictions_scores, labels_keypoints, dist_thresh=5):
    """
    Compute precision, recall metrics and mAP for predicted keypoints against ground truth.

    Args:
        predictions_keypoints (numpy.ndarray): Predicted keypoints of shape (N, 2).
        predictions_scores (numpy.ndarray): Confidence scores of shape (N,).
        labels_keypoints (numpy.ndarray): Ground truth keypoints of shape (M, 2).
        dist_thresh (float): Distance threshold for matching keypoints.

    Returns:
        dict: Dictionary containing:
            - Average Precision (float)
            - Recall at 0.5 (float)
            - Precision at 0.5 (float)
    """
    precision, recall = precision_recall_vectors(
        predictions_keypoints=predictions_keypoints,
        predictions_scores=predictions_scores,
        labels_keypoints=labels_keypoints,
        dist_thresh=dist_thresh
    )

    ap = average_precision(precision, recall)
    return {"ap": ap, "recall": recall[-2], "precision": precision[-2]}
