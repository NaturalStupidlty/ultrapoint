"""util functions
# many old functions, need to clean up
# homography --> homography
# warping
# loss --> delete if useless
"""


import os
import numpy
import torch
import torch.nn.functional as F
import torch.nn as nn
import datetime

from ultrapoint.utils.d2s import DepthToSpace, SpaceToDepth
from ultrapoint.utils.torch_helpers import torch_to_numpy


def saveImg(img, filename):
    cv2.imwrite(filename, img)


def homography_scaling_torch(homography, H, W):
    trans = torch.tensor([[2.0 / W, 0.0, -1], [0.0, 2.0 / H, -1], [0.0, 0.0, 1.0]])
    homography = trans.inverse() @ homography @ trans
    return homography


def filter_points(points, shape, return_mask=False):
    points = points.float()
    shape = shape.float()
    mask = (points >= 0) * (points <= shape - 1)
    mask = torch.prod(mask, dim=-1) == 1
    if return_mask:
        return points[mask], mask
    return points[mask]


def warp_points(points, homographies, device="cpu"):
    """
    Warp a list of points with the given homography.

    Arguments:
        points: list of N points, shape (N, 2(x, y))).
        homography: batched or not (shapes (B, 3, 3) and (...) respectively).

    Returns: a Tensor of shape (N, 2) or (B, N, 2(x, y)) (depending on whether the homography
            is batched) containing the new coordinates of the warped points.

    """
    # expand points len to (x, y, 1)
    no_batches = len(homographies.shape) == 2
    homographies = homographies.unsqueeze(0) if no_batches else homographies
    # homographies = homographies.unsqueeze(0) if len(homographies.shape) == 2 else homographies
    batch_size = homographies.shape[0]
    points = torch.cat(
        (points.float(), torch.ones((points.shape[0], 1)).to(device)), dim=1
    )
    points = points.to(device)
    homographies = homographies.view(batch_size * 3, 3)
    # warped_points = homographies*points
    # points = points.double()
    warped_points = homographies @ points.transpose(0, 1)
    # warped_points = numpy.tensordot(homographies, points.transpose(), axes=([2], [0]))
    # normalize the points
    warped_points = warped_points.view([batch_size, 3, -1])
    warped_points = warped_points.transpose(2, 1)
    warped_points = warped_points[:, :, :2] / warped_points[:, :, 2:]
    return warped_points[0, :, :] if no_batches else warped_points


# from utils.utils import inv_warp_image_batch
def inv_warp_image_batch(img, mat_homo_inv, device="cpu", mode="bilinear"):
    """
    Inverse warp images in batch

    :param img:
        batch of images
        tensor [batch_size, 1, H, W]
    :param mat_homo_inv:
        batch of homography matrices
        tensor [batch_size, 3, 3]
    :param device:
        GPU device or CPU
    :return:
        batch of warped images
        tensor [batch_size, 1, H, W]
    """
    # compute inverse warped points
    if len(img.shape) == 2 or len(img.shape) == 3:
        img = img.view(1, 1, img.shape[0], img.shape[1])
    if len(mat_homo_inv.shape) == 2:
        mat_homo_inv = mat_homo_inv.view(1, 3, 3)

    Batch, channel, H, W = img.shape
    coor_cells = torch.stack(
        torch.meshgrid(
            torch.linspace(-1, 1, W), torch.linspace(-1, 1, H), indexing="ij"
        ),
        dim=2,
    )
    coor_cells = coor_cells.transpose(0, 1)
    coor_cells = coor_cells.to(device)
    coor_cells = coor_cells.contiguous()

    src_pixel_coords = warp_points(coor_cells.view([-1, 2]), mat_homo_inv, device)
    src_pixel_coords = src_pixel_coords.view([Batch, H, W, 2])
    src_pixel_coords = src_pixel_coords.float()

    warped_img = F.grid_sample(img, src_pixel_coords, mode=mode, align_corners=True)
    return warped_img


def inv_warp_image(img, mat_homo_inv, device="cpu", mode="bilinear"):
    """
    Inverse warp images in batch

    :param img:
        batch of images
        tensor [H, W]
    :param mat_homo_inv:
        batch of homography matrices
        tensor [3, 3]
    :param device:
        GPU device or CPU
    :return:
        batch of warped images
        tensor [H, W]
    """
    warped_img = inv_warp_image_batch(img, mat_homo_inv, device, mode)
    return warped_img.squeeze()


def labels2Dto3D(labels, cell_size, add_dustbin=True):
    """
    Change the shape of labels into 3D. Batch of labels.

    :param labels:
        tensor [batch_size, 1, H, W]
        keypoint map.
    :param cell_size:
        8
    :return:
         labels: tensors[batch_size, 65, Hc, Wc]
    """
    batch_size, channel, H, W = labels.shape
    Hc, Wc = H // cell_size, W // cell_size
    space2depth = SpaceToDepth(8)
    labels = space2depth(labels)
    if add_dustbin:
        dustbin = labels.sum(dim=1)
        dustbin = 1 - dustbin
        dustbin[dustbin < 1.0] = 0
        # print('dust: ', dustbin.shape)
        # labels = torch.cat((labels, dustbin.view(batch_size, 1, Hc, Wc)), dim=1)
        labels = torch.cat((labels, dustbin.view(batch_size, 1, Hc, Wc)), dim=1)
        ## norm
        dn = labels.sum(dim=1)
        labels = labels.div(torch.unsqueeze(dn, 1))
    return labels


def labels2Dto3D_flattened(labels, cell_size):
    """
    Change the shape of labels into 3D. Batch of labels.

    :param labels:
        tensor [batch_size, 1, H, W]
        keypoint map.
    :param cell_size:
        8
    :return:
         labels: tensors[batch_size, 65, Hc, Wc]
    """
    batch_size, channel, H, W = labels.shape
    Hc, Wc = H // cell_size, W // cell_size
    space2depth = SpaceToDepth(8)
    # labels = space2depth(labels).squeeze(0)
    labels = space2depth(labels)
    # print("labels in 2Dto3D: ", labels.shape)
    # labels = labels.view(batch_size, H, 1, W, 1)
    # labels = labels.view(batch_size, Hc, cell_size, Wc, cell_size)
    # labels = labels.transpose(1, 2).transpose(3, 4).transpose(2, 3)
    # labels = labels.reshape(batch_size, 1, cell_size ** 2, Hc, Wc)
    # labels = labels.view(batch_size, cell_size ** 2, Hc, Wc)

    dustbin = torch.ones((batch_size, 1, Hc, Wc)).cuda()
    # labels = torch.cat((labels, dustbin.view(batch_size, 1, Hc, Wc)), dim=1)
    labels = torch.cat((labels * 2, dustbin.view(batch_size, 1, Hc, Wc)), dim=1)
    labels = torch.argmax(labels, dim=1)
    return labels


def flattenDetection(detector_output):
    """
    Flatten detection output

    :param detector_output:
        output from detector head
        tensor [65, Hc, Wc]
        :or
        tensor (batch_size, 65, Hc, Wc)

    :return:
        3D heatmap
        numpy (1, H, C)
        :or
        tensor (batch_size, 65, Hc, Wc)

    """
    dense = nn.functional.softmax(detector_output, dim=1)  # [batch, 65, Hc, Wc]
    # Remove dustbin.
    nodust = dense[:, :-1, :, :]
    # Reshape to get full resolution heatmap.
    # heatmap = flatten64to1(nodust, tensor=True) # [1, H, W]
    depth2space = DepthToSpace(8)
    heatmap = depth2space(nodust)

    return heatmap


import cv2


def getPtsFromHeatmap(heatmap, conf_thresh, nms_dist):
    """
    :param self:
    :param heatmap:
        numpy (H, W)
    :return:
    """
    border_remove = 4

    H, W = heatmap.shape[0], heatmap.shape[1]
    xs, ys = numpy.where(heatmap >= conf_thresh)  # Confidence threshold.
    sparsemap = heatmap >= conf_thresh
    if len(xs) == 0:
        return numpy.zeros((3, 0))
    pts = numpy.zeros((3, len(xs)))  # Populate point data sized 3xN.
    pts[0, :] = ys
    pts[1, :] = xs
    pts[2, :] = heatmap[xs, ys]
    pts, _ = nms_fast(pts, H, W, dist_thresh=nms_dist)  # Apply NMS.
    inds = numpy.argsort(pts[2, :])
    pts = pts[:, inds[::-1]]  # Sort by confidence.
    # Remove points along border.
    bord = border_remove
    toremoveW = numpy.logical_or(pts[0, :] < bord, pts[0, :] >= (W - bord))
    toremoveH = numpy.logical_or(pts[1, :] < bord, pts[1, :] >= (H - bord))
    toremove = numpy.logical_or(toremoveW, toremoveH)
    pts = pts[:, ~toremove]
    return pts


def box_nms(prob, size, iou=0.1, min_prob=0.01, keep_top_k=0):
    # requires https://github.com/open-mmlab/mmdetection.
    # Warning : BUILD FROM SOURCE using command MMCV_WITH_OPS=1 pip install -e
    # from mmcv.ops import nms as nms_mmdet
    from torchvision.ops import nms

    """Performs non maximum suppression on the heatmap by considering hypothetical
    bounding boxes centered at each pixel's location (e.g. corresponding to the receptive
    field). Optionally only keeps the top k detections.
    Arguments:
    prob: the probability heatmap, with shape `[H, W]`.
    size: a scalar, the size of the bouding boxes.
    iou: a scalar, the IoU overlap threshold.
    min_prob: a threshold under which all probabilities are discarded before NMS.
    keep_top_k: an integer, the number of top scores to keep.
    """
    pts = torch.nonzero(prob > min_prob).float()  # [N, 2]
    prob_nms = torch.zeros_like(prob)
    if pts.nelement() == 0:
        return prob_nms
    size = torch.tensor(size / 2.0).cuda()
    boxes = torch.cat([pts - size, pts + size], dim=1)  # [N, 4]
    scores = prob[pts[:, 0].long(), pts[:, 1].long()]
    if keep_top_k != 0:
        indices = nms(boxes, scores, iou)
    else:
        raise NotImplementedError
        # indices, _ = nms(boxes, scores, iou, boxes.size()[0])
        # print("boxes: ", boxes.shape)
        # print("scores: ", scores.shape)
        # proposals = torch.cat([boxes, scores.unsqueeze(-1)], dim=-1)
        # dets, indices = nms_mmdet(proposals, iou)
        # indices = indices.long()

        # indices = box_nms_retinaNet(boxes, scores, iou)
    pts = torch.index_select(pts, 0, indices)
    scores = torch.index_select(scores, 0, indices)
    prob_nms[pts[:, 0].long(), pts[:, 1].long()] = scores
    return prob_nms


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
    grid = numpy.zeros((H, W)).astype(int)  # Track NMS data.
    inds = numpy.zeros((H, W)).astype(int)  # Store indices of points.
    # Sort by confidence and round to nearest int.
    inds1 = numpy.argsort(-in_corners[2, :])
    corners = in_corners[:, inds1]
    rcorners = corners[:2, :].round().astype(int)  # Rounded corners.
    # Check for edge case of 0 or 1 corners.
    if rcorners.shape[1] == 0:
        return numpy.zeros((3, 0)).astype(int), numpy.zeros(0).astype(int)
    if rcorners.shape[1] == 1:
        out = numpy.vstack((rcorners, in_corners[2])).reshape(3, 1)
        return out, numpy.zeros((1)).astype(int)
    # Initialize the grid.
    for i, rc in enumerate(rcorners.T):
        grid[rcorners[1, i], rcorners[0, i]] = 1
        inds[rcorners[1, i], rcorners[0, i]] = i
    # Pad the border of the grid, so that we can NMS points near the border.
    pad = dist_thresh
    grid = numpy.pad(grid, ((pad, pad), (pad, pad)), mode="constant")
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
    keepy, keepx = numpy.where(grid == -1)
    keepy, keepx = keepy - pad, keepx - pad
    inds_keep = inds[keepy, keepx]
    out = corners[:, inds_keep]
    values = out[-1, :]
    inds2 = numpy.argsort(-values)
    out = out[:, inds2]
    out_inds = inds1[inds_keep[inds2]]
    return out, out_inds


def compute_mask(image_shape, inv_homography, device="cpu", erosion_radius=0):
    """
    Compute a boolean mask of the valid pixels resulting from an homography applied to
    an image of a given shape. Pixels that are False correspond to bordering artifacts.
    A margin can be discarded using erosion.

    Arguments:
        input_shape: Tensor of rank 2 representing the image shape, i.e. `[H, W]`.
        homography: Tensor of shape (B, 8) or (8,), where B is the batch size.
        `erosion_radius: radius of the margin to be discarded.

    Returns: a Tensor of type `tf.int32` and shape (H, W).
    """
    # mask = H_transform(tf.ones(image_shape), homography, interpolation='NEAREST')
    # mask = H_transform(tf.ones(image_shape), homography, interpolation='NEAREST')
    if inv_homography.dim() == 2:
        inv_homography = inv_homography.view(-1, 3, 3)
    batch_size = inv_homography.shape[0]
    mask = torch.ones(batch_size, 1, image_shape[0], image_shape[1]).to(device)
    mask = inv_warp_image_batch(mask, inv_homography, device=device, mode="nearest")
    mask = mask.view(batch_size, image_shape[0], image_shape[1])
    mask = mask.cpu().numpy()
    if erosion_radius > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erosion_radius * 2,) * 2)
        for i in range(batch_size):
            mask[i, :, :] = cv2.erode(mask[i, :, :], kernel, iterations=1)

    return torch.tensor(mask).to(device)


def normPts(pts, shape):
    """
    normalize pts to [-1, 1]
    :param pts:
        tensor (y, x)
    :param shape:
        tensor shape (y, x)
    :return:
    """
    pts = pts / shape * 2 - 1
    return pts


def denormPts(pts, shape):
    """
    denormalize pts back to H, W
    :param pts:
        tensor (y, x)
    :param shape:
        numpy (y, x)
    :return:
    """
    pts = (pts + 1) * shape / 2
    return pts


def descriptor_loss(
    descriptors,
    descriptors_warped,
    homographies,
    mask_valid=None,
    cell_size=8,
    lamda_d=250,
    device="cpu",
    descriptor_dist=4,
    **config
):
    """
    Compute descriptor loss from descriptors_warped and given homographies

    :param descriptors:
        Output from descriptor head
        tensor [batch_size, descriptors, Hc, Wc]
    :param descriptors_warped:
        Output from descriptor head of warped image
        tensor [batch_size, descriptors, Hc, Wc]
    :param homographies:
        known homographies
    :param cell_size:
        8
    :param device:
        gpu or cpu
    :param config:
    :return:
        loss, and other tensors for visualization
    """

    # put to gpu
    homographies = homographies.to(device)
    # config
    from src.ultrapoint.utils.utils import warp_points

    lamda_d = lamda_d  # 250
    margin_pos = 1
    margin_neg = 0.2
    batch_size, Hc, Wc = (
        descriptors.shape[0],
        descriptors.shape[2],
        descriptors.shape[3],
    )
    #####
    # H, W = Hc.numpy().astype(int) * cell_size, Wc.numpy().astype(int) * cell_size
    H, W = Hc * cell_size, Wc * cell_size
    #####
    with torch.no_grad():
        # shape = torch.tensor(list(descriptors.shape[2:]))*torch.tensor([cell_size, cell_size]).type(torch.FloatTensor).to(device)
        shape = torch.tensor([H, W]).type(torch.FloatTensor).to(device)
        # compute the center pixel of every cell in the image

        coor_cells = torch.stack(
            torch.meshgrid(torch.arange(Hc), torch.arange(Wc), indexing="ij"), dim=2
        )
        coor_cells = coor_cells.type(torch.FloatTensor).to(device)
        coor_cells = coor_cells * cell_size + cell_size // 2
        ## coord_cells is now a grid containing the coordinates of the Hc x Wc
        ## center pixels of the 8x8 cells of the image

        # coor_cells = coor_cells.view([-1, Hc, Wc, 1, 1, 2])
        coor_cells = coor_cells.view([-1, 1, 1, Hc, Wc, 2])  # be careful of the order
        # warped_coor_cells = warp_points(coor_cells.view([-1, 2]), homographies, device)
        warped_coor_cells = normPts(coor_cells.view([-1, 2]), shape)
        warped_coor_cells = torch.stack(
            (warped_coor_cells[:, 1], warped_coor_cells[:, 0]), dim=1
        )  # (y, x) to (x, y)
        warped_coor_cells = warp_points(warped_coor_cells, homographies, device)

        warped_coor_cells = torch.stack(
            (warped_coor_cells[:, :, 1], warped_coor_cells[:, :, 0]), dim=2
        )  # (batch, x, y) to (batch, y, x)

        shape_cell = (
            torch.tensor([H // cell_size, W // cell_size])
            .type(torch.FloatTensor)
            .to(device)
        )
        # warped_coor_mask = denormPts(warped_coor_cells, shape_cell)

        warped_coor_cells = denormPts(warped_coor_cells, shape)
        # warped_coor_cells = warped_coor_cells.view([-1, 1, 1, Hc, Wc, 2])
        warped_coor_cells = warped_coor_cells.view([-1, Hc, Wc, 1, 1, 2])
        #     print("warped_coor_cells: ", warped_coor_cells.shape)
        # compute the pairwise distance
        cell_distances = coor_cells - warped_coor_cells
        cell_distances = torch.norm(cell_distances, dim=-1)
        ##### check
        #     print("descriptor_dist: ", descriptor_dist)
        mask = cell_distances <= descriptor_dist  # 0.5 # trick

        mask = mask.type(torch.FloatTensor).to(device)

    # compute the pairwise dot product between descriptors: d^t * d
    descriptors = descriptors.transpose(1, 2).transpose(2, 3)
    descriptors = descriptors.view((batch_size, Hc, Wc, 1, 1, -1))
    descriptors_warped = descriptors_warped.transpose(1, 2).transpose(2, 3)
    descriptors_warped = descriptors_warped.view((batch_size, 1, 1, Hc, Wc, -1))
    dot_product_desc = descriptors * descriptors_warped
    dot_product_desc = dot_product_desc.sum(dim=-1)
    ## dot_product_desc.shape = [batch_size, Hc, Wc, Hc, Wc, desc_len]

    # hinge loss
    positive_dist = torch.max(
        margin_pos - dot_product_desc, torch.tensor(0.0).to(device)
    )
    # positive_dist[positive_dist < 0] = 0
    negative_dist = torch.max(
        dot_product_desc - margin_neg, torch.tensor(0.0).to(device)
    )
    # negative_dist[neative_dist < 0] = 0
    # sum of the dimension

    if mask_valid is None:
        # mask_valid = torch.ones_like(mask)
        mask_valid = torch.ones(batch_size, 1, Hc * cell_size, Wc * cell_size)
    mask_valid = mask_valid.view(
        batch_size, 1, 1, mask_valid.shape[2], mask_valid.shape[3]
    )

    loss_desc = lamda_d * mask * positive_dist + (1 - mask) * negative_dist
    loss_desc = loss_desc * mask_valid
    # mask_validg = torch.ones_like(mask)
    ##### bug in normalization
    normalization = batch_size * (mask_valid.sum() + 1) * Hc * Wc
    pos_sum = (lamda_d * mask * positive_dist / normalization).sum()
    neg_sum = ((1 - mask) * negative_dist / normalization).sum()
    loss_desc = loss_desc.sum() / normalization
    # loss_desc = loss_desc.sum() / (batch_size * Hc * Wc)
    # return loss_desc, mask, mask_valid, positive_dist, negative_dist
    return loss_desc, mask, pos_sum, neg_sum


def prepare_experiment_directory(
    experiment_directory: str, experiment_name: str, date: bool = True
) -> str:
    if date:
        postfix = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    else:
        postfix = ""

    output_directory = os.path.join(
        experiment_directory, experiment_name + "_" + postfix
    )
    os.makedirs(output_directory, exist_ok=True)

    return output_directory


def crop_or_pad_choice(in_num_points, out_num_points, shuffle=False):
    # Adapted from https://github.com/haosulab/frustum_pointnet/blob/635c938f18b9ec1de2de717491fb217df84d2d93/fpointnet/data/datasets/utils.py
    """Crop or pad point cloud to a fixed number; return the indexes
    Args:
        points (numpy.ndarray): point cloud. (n, d)
        num_points (int): the number of output points
        shuffle (bool): whether to shuffle the order
    Returns:
        numpy.ndarray: output point cloud
        numpy.ndarray: index to choose input points
    """
    if shuffle:
        choice = numpy.random.permutation(in_num_points)
    else:
        choice = numpy.arange(in_num_points)
    assert out_num_points > 0, (
        "out_num_points = %d must be positive int!" % out_num_points
    )
    if in_num_points >= out_num_points:
        choice = choice[:out_num_points]
    else:
        num_pad = out_num_points - in_num_points
        pad = numpy.random.choice(choice, num_pad, replace=True)
        choice = numpy.concatenate([choice, pad])
    return choice


def mask_to_keypoints(mask):
    """
    Convert a binary mask tensor of shape (1,1,H,W) (values 0/1)
    into an (n,2) numpy array of [x, y] keypoint coordinates.
    """
    # squeeze down to (H,W)
    if isinstance(mask, torch.Tensor):
        m = mask.squeeze().cpu().detach().numpy()
    else:
        m = mask.squeeze()
    ys, xs = numpy.nonzero(m)
    # stack into [[x1,y1], [x2,y2], ...]
    return numpy.stack([xs, ys], axis=1)
