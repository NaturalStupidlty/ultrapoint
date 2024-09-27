import argparse
import dotenv
import os

from pathlib import Path

import numpy as np
from imageio import imread
from tqdm import tqdm
from tensorboardX import SummaryWriter

import torch
import torch.optim
import torch.utils.data

from utils.utils import inv_warp_image_batch
from utils.utils import prepare_experiment_directory
from utils.logging import create_logger, logger, log_data_size
from utils.loader import dataLoader_test as dataLoader
from utils.utils import saveImg
from utils.loader import get_module
from utils.draw import draw_keypoints
from utils.config_helpers import load_config, save_config
from utils.torch_helpers import make_deterministic, set_precision, determine_device
from models.model_wrap import SuperPointFrontend_torch, PointTracker
from utils.loader import get_checkpoints_path
from utils.var_dim import squeezeToNumpy


def combine_heatmap(heatmap, inv_homographies, mask_2d, device="cpu"):
    heatmap = heatmap * mask_2d
    heatmap = inv_warp_image_batch(
        heatmap, inv_homographies[0, :, :, :], device=device, mode="bilinear"
    )

    mask_2d = inv_warp_image_batch(
        mask_2d, inv_homographies[0, :, :, :], device=device, mode="bilinear"
    )
    heatmap = torch.sum(heatmap, dim=0)
    mask_2d = torch.sum(mask_2d, dim=0)

    return heatmap / mask_2d


def export_descriptor(config, output_directory, args):
    """
    # input 2 images, output keypoints and correspondence
    save prediction:
        pred:
            'image': np(320,240)
            'prob' (keypoints): np (N1, 2)
            'desc': np (N2, 256)
            'warped_image': np(320,240)
            'warped_prob' (keypoints): np (N2, 2)
            'warped_desc': np (N2, 256)
            'homography': np (3,3)
            'matches': np [N3, 4]
    """
    device = determine_device()
    logger.info(f"Training with device: {device}")
    save_config(os.path.join(output_directory, "config.yaml"), config)

    save_output = os.path.join(get_checkpoints_path(output_directory), "predictions")
    os.makedirs(save_output, exist_ok=True)

    ## parameters
    outputMatches = True
    subpixel = config["model"]["subpixel"]["enable"]
    patch_size = config["model"]["subpixel"]["patch_size"]

    # data loading
    data = dataLoader(config, dataset=config["data"]["dataset"])
    test_set, test_loader = data["test_set"], data["test_loader"]
    log_data_size(test_loader, config, tag="test")

    # model loading
    Val_model_heatmap = get_module(config["front_end_model"])
    val_agent = Val_model_heatmap(config["model"], device=device)
    val_agent.loadModel()

    # writer from tensorboard
    val_agent.writer = SummaryWriter(output_directory)

    ## tracker
    tracker = PointTracker(max_length=2, nn_thresh=val_agent.nn_thresh)

    for i, sample in tqdm(enumerate(test_loader)):
        img_0, img_1 = sample["image"], sample["warped_image"]

        # first image, no matches
        # img = img_0
        def get_pts_desc_from_agent(val_agent, img, device="cpu"):
            """
            pts: list [numpy (3, N)]
            desc: list [numpy (256, N)]
            """
            heatmap_batch = val_agent.run(
                img.to(device)
            )  # heatmap: numpy [batch, 1, H, W]
            # heatmap to pts
            pts = val_agent.heatmap_to_pts()
            # print("pts: ", pts)
            if subpixel:
                pts = val_agent.soft_argmax_points(pts, patch_size=patch_size)
            # heatmap, pts to desc
            desc_sparse = val_agent.desc_to_sparseDesc()
            # print("pts[0]: ", pts[0].shape, ", desc_sparse[0]: ", desc_sparse[0].shape)
            # print("pts[0]: ", pts[0].shape)
            outs = {"pts": pts[0], "desc": desc_sparse[0]}
            return outs

        def transpose_np_dict(outs):
            for entry in list(outs):
                outs[entry] = outs[entry].transpose()

        outs = get_pts_desc_from_agent(val_agent, img_0, device=device)
        pts, desc = outs["pts"], outs["desc"]  # pts: np [3, N]

        if outputMatches == True:
            tracker.update(pts, desc)

        # save keypoints
        pred = {"image": squeezeToNumpy(img_0)}
        pred.update({"prob": pts.transpose(), "desc": desc.transpose()})

        # second image, output matches
        outs = get_pts_desc_from_agent(val_agent, img_1, device=device)
        pts, desc = outs["pts"], outs["desc"]

        if outputMatches == True:
            tracker.update(pts, desc)

        pred.update({"warped_image": squeezeToNumpy(img_1)})
        # print("total points: ", pts.shape)
        pred.update(
            {
                "warped_prob": pts.transpose(),
                "warped_desc": desc.transpose(),
                "homography": squeezeToNumpy(sample["homography"]),
            }
        )

        if outputMatches == True:
            matches = tracker.get_matches()
            print("matches: ", matches.transpose().shape)
            pred.update({"matches": matches.transpose()})
        print("pts: ", pts.shape, ", desc: ", desc.shape)

        tracker.clear_desc()

        path = os.path.join(save_output, f"{i}.npz")
        np.savez_compressed(path, **pred)
        logger.debug(f"save: {path}")

    logger.info("output pairs: {len(test_loader)}")


@torch.no_grad()
def export_detector_homo_adapt_gpu(config, output_directory, args):
    """
    input 1 images, output pseudo ground truth by homography adaptation.
    Save labels:
        pred:
            'prob' (keypoints): np (N1, 3)
    """
    # basic setting
    task = config["data"]["dataset"]
    export_task = config["data"]["export_folder"]
    device = determine_device()
    logger.info(f"Training with device: {device}")
    save_config(os.path.join(output_directory, "config.yaml"), config)

    ## parameters
    nms_dist = config["model"]["nms"]  # 4
    top_k = config["model"]["top_k"]
    homoAdapt_iter = config["data"]["homography_adaptation"]["num"]
    conf_thresh = config["model"]["detection_threshold"]
    nn_thresh = 0.7
    outputMatches = True
    count = 0
    max_length = 5
    output_images = args.outputImg
    check_exist = True

    ## save data
    save_path = Path(output_directory)
    save_output = save_path
    save_output = save_output / "predictions" / export_task
    save_path = save_path / "checkpoints"
    logger.info("=> will save everything to {}".format(save_path))
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(save_output, exist_ok=True)

    # data loading
    from utils.loader import dataLoader_test as dataLoader

    data = dataLoader(config, dataset=task, export_task=export_task)
    test_set, test_loader = data["test_set"], data["test_loader"]

    # model loading
    ## load pretrained
    try:
        path = config["pretrained"]
        print("==> Loading pre-trained network.")
        print("path: ", path)
        # This class runs the SuperPoint network and processes its outputs.

        fe = SuperPointFrontend_torch(
            config=config,
            weights_path=path,
            nms_dist=nms_dist,
            conf_thresh=conf_thresh,
            nn_thresh=nn_thresh,
            cuda=False,
            device=device,
        )
        print("==> Successfully loaded pre-trained network.")

        fe.net_parallel()
        print(path)
        # save to files
        save_file = save_output / "export.txt"
        with open(save_file, "a") as myfile:
            myfile.write("load model: " + path + "\n")
    except Exception:
        print(f"load model: {path} failed! ")
        raise

    def load_as_float(path):
        return imread(path).astype(np.float32) / 255

    tracker = PointTracker(max_length, nn_thresh=fe.nn_thresh)
    with open(save_file, "a") as myfile:
        myfile.write("homography adaptation: " + str(homoAdapt_iter) + "\n")

    ## loop through all images
    for i, sample in tqdm(enumerate(test_loader)):
        img, mask_2D = sample["image"], sample["valid_mask"]
        img = img.transpose(0, 1)
        img_2D = sample["image_2D"].numpy().squeeze()
        mask_2D = mask_2D.transpose(0, 1)

        inv_homographies, homographies = (
            sample["homographies"],
            sample["inv_homographies"],
        )
        img, mask_2D, homographies, inv_homographies = (
            img.to(device),
            mask_2D.to(device),
            homographies.to(device),
            inv_homographies.to(device),
        )
        # sample = test_set[i]
        name = sample["name"][0]
        logger.info(f"name: {name}")
        if check_exist:
            p = Path(save_output, "{}.npz".format(name))
            if p.exists():
                logger.info("file %s exists. skip the sample.", name)
                continue

        # pass through network
        heatmap = fe.run(img, onlyHeatmap=True, train=False)
        outputs = combine_heatmap(heatmap, inv_homographies, mask_2D, device=device)
        pts = fe.getPtsFromHeatmap(outputs.detach().cpu().squeeze())  # (x,y, prob)

        # subpixel prediction
        if config["model"]["subpixel"]["enable"]:
            fe.heatmap = outputs  # tensor [batch, 1, H, W]
            print("outputs: ", outputs.shape)
            print("pts: ", pts.shape)
            pts = fe.soft_argmax_points([pts])
            pts = pts[0]

        ## top K points
        pts = pts.transpose()
        print("total points: ", pts.shape)
        print("pts: ", pts[:5])
        if top_k:
            if pts.shape[0] > top_k:
                pts = pts[:top_k, :]
                print("topK filter: ", pts.shape)

        ## save keypoints
        pred = {}
        pred.update({"pts": pts})

        ## - make directories
        filename = str(name)
        if task == "Kitti" or "Kitti_inh":
            scene_name = sample["scene_name"][0]
            os.makedirs(Path(save_output, scene_name), exist_ok=True)

        path = Path(save_output, "{}.npz".format(filename))
        np.savez_compressed(path, **pred)

        ## output images for visualization labels
        if output_images:
            img_pts = draw_keypoints(img_2D * 255, pts.transpose())
            f = save_output / (str(count) + ".png")
            if task == "Coco" or "Kitti":
                f = save_output / (name + ".png")
            saveImg(img_pts, str(f))
        count += 1

    print("output pseudo ground truth: ", count)
    save_file = save_output / "export.txt"
    with open(save_file, "a") as myfile:
        myfile.write("Homography adaptation: " + str(homoAdapt_iter) + "\n")
        myfile.write("output pairs: " + str(count) + "\n")
    pass


def parse_arguments():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")

    # export command
    p_train = subparsers.add_parser("export_descriptor")
    p_train.add_argument("config", type=str)
    p_train.add_argument("exper_name", type=str)
    p_train.add_argument("--correspondence", action="store_true")
    p_train.add_argument("--eval", action="store_true")
    p_train.add_argument(
        "--debug", action="store_true", default=False, help="turn on debuging mode"
    )
    p_train.set_defaults(func=export_descriptor)

    # using homography adaptation to export detection psuedo ground truth
    p_train = subparsers.add_parser("export_detector_homo_adapt_gpu")
    p_train.add_argument("config", type=str)
    p_train.add_argument("exper_name", type=str)
    p_train.add_argument("--eval", action="store_true")
    p_train.add_argument(
        "--outputImg", action="store_true", help="output image for visualization"
    )
    p_train.add_argument(
        "--debug", action="store_true", default=False, help="turn on debuging mode"
    )
    p_train.set_defaults(func=export_detector_homo_adapt_gpu)

    return parser.parse_args()


def main():
    dotenv.load_dotenv()
    args = parse_arguments()
    config = load_config(args.config)
    set_precision(config["precision"])
    make_deterministic(config["seed"])
    output_directory = prepare_experiment_directory(
        os.getenv("EXPER_PATH"), args.exper_name
    )
    create_logger(**config["logging"], logs_dir=output_directory)
    args.func(config, output_directory)


if __name__ == "__main__":
    main()
