"""
Adapted from https://github.com/rpautrat/SuperPoint/blob/master/superpoint/datasets/synthetic_dataset.py

Author: You-Yi Jau, Rui Zhu
Date: 2019/12/12
"""

import cv2
import shutil
import tarfile
import multiprocessing
import torch.utils.data as data
import torch
import numpy as np

from typing import Union, List
from imageio import imread
from tqdm import tqdm
from loguru import logger
from pathlib import Path

from src.ultrapoint.datasets import synthetic_dataset
from src.ultrapoint.utils.homographies import (
    sample_homography as sample_homography,
)
from src.ultrapoint.utils.photometric import (
    ImgAugTransform,
    customizedTransform,
)
from src.ultrapoint.utils.utils import compute_valid_mask
from src.ultrapoint.utils.utils import inv_warp_image, warp_points


def load_as_float(path):
    return imread(path).astype(np.float32) / 255


class SyntheticDatasetGaussian(data.Dataset):
    DRAWING_PRIMITIVES = [
        "draw_lines",
        "draw_polygon",
        "draw_multiple_polygons",
        "draw_ellipses",
        "draw_star",
        "draw_checkerboard",
        "draw_stripes",
        "draw_cube",
        "gaussian_noise",
    ]

    def __init__(
        self,
        task="train",
        transforms=None,
        **config,
    ):
        self.config = config
        self._data_path = config.get("path", "/tmp")
        self.transform = transforms

        self.enable_photo_train = config["augmentation"]["photometric"]["enable_train"]
        self.enable_homo_train = config["augmentation"]["homographic"]["enable_train"]
        self.enable_homo_val = config["augmentation"]["homographic"]["enable_val"]
        self.enable_photo_val = config["augmentation"]["photometric"]["enable_val"]

        self.action = "training" if task == "train" else "validation"
        self.config["photometric"]["enable"] = config["photometric"][f"enable_{task}"]
        self.gaussian_label = config["gaussian_label"]["enable"]
        self.pool = multiprocessing.Pool(6)

        # Parse drawing primitives
        primitives = self.setup_primitives(config["primitives"])
        logger.info(primitives)

        basepath = Path(
            self._data_path,
            "synthetic_shapes"
            + ("_{}".format(config["suffix"]) if config["suffix"] is not None else ""),
        )
        basepath.mkdir(parents=True, exist_ok=True)

        splits = {s: {"images": [], "points": []} for s in [self.action]}
        for primitive in primitives:
            tar_path = Path(basepath, f"{primitive}.tar.gz")
            if not tar_path.exists():
                self.dump_primitive_data(primitive, tar_path, config)

            # Untar locally
            logger.debug(f"Extracting archive for primitive {primitive}")
            logger.debug(f"tar_path: {tar_path}")
            tar = tarfile.open(tar_path)
            temp_dir = Path(self._data_path)
            tar.extractall(path=temp_dir)
            tar.close()

            # Gather filenames in all splits, optionally truncate
            truncate = config["truncate"].get(primitive, 1)
            path = Path(temp_dir, primitive)
            for s in splits:
                e = [str(p) for p in Path(path, "images", s).iterdir()]
                f = [p.replace("images", "points") for p in e]
                f = [p.replace(".png", ".npy") for p in f]
                splits[s]["images"].extend(e[: int(truncate * len(e))])
                splits[s]["points"].extend(f[: int(truncate * len(f))])

        # Shuffle
        for s in splits:
            perm = np.random.RandomState(0).permutation(len(splits[s]["images"]))
            for obj in ["images", "points"]:
                splits[s][obj] = np.array(splits[s][obj])[perm].tolist()

        self.crawl_folders(splits)

    def __getitem__(self, index):
        """
        :param index:
        :return:
            labels_2D: tensor(1, H, W)
            image: tensor(1, H, W)
        """

        def imgPhotometric(img):
            """

            :param img:
                numpy (H, W)
            :return:
            """
            augmentation = ImgAugTransform(**self.config["augmentation"])
            img = img[:, :, np.newaxis]
            img = augmentation(img)
            cusAug = customizedTransform()
            img = cusAug(img, **self.config["augmentation"])
            return img

        def get_labels(pnts, H, W):
            labels = torch.zeros(H, W)
            # print('--2', pnts, pnts.size())
            # pnts_int = torch.min(pnts.round().long(), torch.tensor([[H-1, W-1]]).long())
            pnts_int = torch.min(
                pnts.round().long(), torch.tensor([[W - 1, H - 1]]).long()
            )
            # print('--3', pnts_int, pnts_int.size())
            labels[pnts_int[:, 1], pnts_int[:, 0]] = 1
            return labels

        def get_label_res(H, W, pnts):
            quan = lambda x: x.round().long()
            labels_res = torch.zeros(H, W, 2)
            # pnts_int = torch.min(pnts.round().long(), torch.tensor([[H-1, W-1]]).long())

            labels_res[quan(pnts)[:, 1], quan(pnts)[:, 0], :] = pnts - pnts.round()
            # print("pnts max: ", quan(pnts).max(dim=0))
            # print("labels_res: ", labels_res.shape)
            labels_res = labels_res.transpose(1, 2).transpose(0, 1)
            return labels_res

        from src.ultrapoint.datasets.data_tools import np_to_tensor
        from src.ultrapoint.utils.utils import filter_points
        from src.ultrapoint.utils.torch_helpers import squeeze_to_numpy

        sample = self.samples[index]
        img = load_as_float(sample["image"])
        H, W = img.shape[0], img.shape[1]
        self.H = H
        self.W = W
        pnts = np.load(sample["points"])  # (y, x)
        pnts = torch.tensor(pnts).float()
        pnts = torch.stack((pnts[:, 1], pnts[:, 0]), dim=1)  # (x, y)
        pnts = filter_points(pnts, torch.tensor([W, H]))
        sample = {}

        # print('pnts: ', pnts[:5])
        # print('--1', pnts)
        labels_2D = get_labels(pnts, H, W)
        sample.update({"labels_2D": labels_2D.unsqueeze(0)})

        if not (
            (
                self.config["augmentation"]["homographic"]["enable_train"]
                and self.action == "training"
            )
            or (
                self.config["augmentation"]["homographic"]["enable_val"]
                and self.action == "validation"
            )
        ):
            img = img[:, :, np.newaxis]
            # labels = labels.view(-1,H,W)
            if self.transform is not None:
                img = self.transform(img)
            sample["image"] = img
            # sample = {'image': img, 'labels_2D': labels}
            valid_mask = compute_valid_mask(
                torch.tensor([H, W]), inv_homography=torch.eye(3)
            )
            sample.update({"valid_mask": valid_mask})
            labels_res = get_label_res(H, W, pnts)
            pnts_post = pnts
            # pnts_for_gaussian = pnts
        else:
            # img_warp = img
            from src.ultrapoint.utils.utils import (
                homography_scaling_torch as homography_scaling,
            )
            from numpy.linalg import inv

            homography = sample_homography(
                np.array([2, 2]),
                shift=-1,
                **self.config["augmentation"]["homographic"]["params"],
            )

            ##### use inverse from the sample homography
            homography = inv(homography)
            ######

            homography = torch.tensor(homography).float()
            inv_homography = homography.inverse()
            img = torch.from_numpy(img)
            warped_img = inv_warp_image(img.squeeze(), inv_homography, mode="bilinear")
            warped_img = warped_img.squeeze().numpy()
            warped_img = warped_img[:, :, np.newaxis]

            # labels = torch.from_numpy(labels)
            # warped_labels = self.inv_warp_image(labels.squeeze(), inv_homography, mode='nearest').unsqueeze(0)
            warped_pnts = warp_points(pnts, homography_scaling(homography, H, W))
            warped_pnts = filter_points(warped_pnts, torch.tensor([W, H]))
            # pnts = warped_pnts[:, [1, 0]]
            # pnts_for_gaussian = warped_pnts
            # warped_labels = torch.zeros(H, W)
            # warped_labels[warped_pnts[:, 1], warped_pnts[:, 0]] = 1
            # warped_labels = warped_labels.view(-1, H, W)

            if self.transform is not None:
                warped_img = self.transform(warped_img)
            # sample = {'image': warped_img, 'labels_2D': warped_labels}
            sample["image"] = warped_img

            valid_mask = compute_valid_mask(
                torch.tensor([H, W]),
                inv_homography=inv_homography,
                erosion_radius=self.config["augmentation"]["homographic"][
                    "valid_border_margin"
                ],
            )  # can set to other value
            sample.update({"valid_mask": valid_mask})

            labels_2D = get_labels(warped_pnts, H, W)
            sample.update({"labels_2D": labels_2D.unsqueeze(0)})

            labels_res = get_label_res(H, W, warped_pnts)
            pnts_post = warped_pnts

        if self.gaussian_label:
            # warped_labels_gaussian = get_labels_gaussian(pnts)
            from src.ultrapoint.datasets.data_tools import get_labels_bi

            labels_2D_bi = get_labels_bi(pnts_post, H, W)

            labels_gaussian = self.gaussian_blur(squeeze_to_numpy(labels_2D_bi))
            labels_gaussian = np_to_tensor(labels_gaussian, H, W)
            sample["labels_2D_gaussian"] = labels_gaussian

        sample.update({"labels_res": labels_res})

        ### code for warped image
        if self.config["warped_pair"]["enable"]:
            from src.ultrapoint.datasets.data_tools import warpLabels

            homography = sample_homography(
                np.array([2, 2]), shift=-1, **self.config["warped_pair"]["params"]
            )

            ##### use inverse from the sample homography
            homography = np.linalg.inv(homography)
            #####
            inv_homography = np.linalg.inv(homography)

            homography = torch.tensor(homography).type(torch.FloatTensor)
            inv_homography = torch.tensor(inv_homography).type(torch.FloatTensor)

            # photometric augmentation from original image

            # warp original image
            warped_img = img.type(torch.FloatTensor)
            warped_img = inv_warp_image(
                warped_img.squeeze(), inv_homography, mode="bilinear"
            ).unsqueeze(0)
            if (self.enable_photo_train == True and self.action == "train") or (
                self.enable_photo_val and self.action == "val"
            ):
                warped_img = imgPhotometric(
                    warped_img.numpy().squeeze()
                )  # numpy array (H, W, 1)
                warped_img = torch.tensor(warped_img, dtype=torch.float32)
                pass
            warped_img = warped_img.view(-1, H, W)

            # warped_labels = warpLabels(pnts, H, W, homography)
            warped_set = warpLabels(pnts, H, W, homography, bilinear=True)
            warped_labels = warped_set["labels"]
            warped_res = warped_set["res"]
            warped_res = warped_res.transpose(1, 2).transpose(0, 1)
            # print("warped_res: ", warped_res.shape)
            if self.gaussian_label:
                # print("do gaussian labels!")
                # warped_labels_gaussian = get_labels_gaussian(warped_set['warped_pnts'].numpy())
                # warped_labels_bi = self.inv_warp_image(labels_2D.squeeze(), inv_homography, mode='nearest').unsqueeze(0) # bilinear, nearest
                warped_labels_bi = warped_set["labels_bi"]
                warped_labels_gaussian = self.gaussian_blur(
                    squeeze_to_numpy(warped_labels_bi)
                )
                warped_labels_gaussian = np_to_tensor(warped_labels_gaussian, H, W)
                sample["warped_labels_gaussian"] = warped_labels_gaussian
                sample.update({"warped_labels_bi": warped_labels_bi})

            sample.update(
                {
                    "warped_img": warped_img,
                    "warped_labels": warped_labels,
                    "warped_res": warped_res,
                }
            )

            # print('erosion_radius', self.config['warped_pair']['valid_border_margin'])
            valid_mask = compute_valid_mask(
                torch.tensor([H, W]),
                inv_homography=inv_homography,
                erosion_radius=self.config["warped_pair"]["valid_border_margin"],
            )  # can set to other value
            sample.update({"warped_valid_mask": valid_mask})
            sample.update(
                {"homographies": homography, "inv_homographies": inv_homography}
            )

        return sample

    def __len__(self):
        return len(self.samples)

    def dump_primitive_data(self, primitive, tar_path, config: dict):
        temp_dir = Path(self._data_path, primitive)

        logger.info(f"Generating .tar file for primitive {primitive}.\n")
        synthetic_dataset.set_random_state(
            np.random.RandomState(config["generation"]["random_seed"])
        )
        for split, size in self.config["generation"]["split_sizes"].items():
            im_dir, pts_dir = [Path(temp_dir, i, split) for i in ["images", "points"]]
            im_dir.mkdir(parents=True, exist_ok=True)
            pts_dir.mkdir(parents=True, exist_ok=True)

            description = f"Generating {primitive} {split} set"
            for i in tqdm(range(size), desc=description, leave=False):
                image = synthetic_dataset.generate_background(
                    config["generation"]["image_size"],
                    **config["generation"]["params"]["generate_background"],
                )
                points = np.array(
                    getattr(synthetic_dataset, primitive)(
                        image, **config["generation"]["params"].get(primitive, {})
                    )
                )
                points = np.flip(points, 1)  # reverse convention with opencv

                b = config["preprocessing"]["blur_size"]
                image = cv2.GaussianBlur(image, (b, b), 0)
                points = (
                    points
                    * np.array(config["preprocessing"]["resize"], float)
                    / np.array(config["generation"]["image_size"], float)
                )
                image = cv2.resize(
                    image,
                    tuple(config["preprocessing"]["resize"][::-1]),
                    interpolation=cv2.INTER_LINEAR,
                )

                cv2.imwrite(str(Path(im_dir, "{}.png".format(i))), image)
                np.save(Path(pts_dir, "{}.npy".format(i)), points)

        # Pack into a tar file
        tar = tarfile.open(tar_path, mode="w:gz")
        tar.add(temp_dir, arcname=primitive)
        tar.close()
        shutil.rmtree(temp_dir)
        logger.info(".tar file dumped to {}.".format(tar_path))

    def setup_primitives(self, names: Union[str, List[str]]):
        if names == "all":
            return self.DRAWING_PRIMITIVES

        return [name for name in names if name in self.DRAWING_PRIMITIVES]

    def crawl_folders(self, splits):
        sequence_set = []
        for img, pnts in zip(
            splits[self.action]["images"], splits[self.action]["points"]
        ):
            sample = {"image": img, "points": pnts}
            sequence_set.append(sample)
        self.samples = sequence_set

    ## util functions
    def gaussian_blur(self, image):
        """
        image: np [H, W]
        return:
            blurred_image: np [H, W]
        """
        aug_par = {"photometric": {}}
        aug_par["photometric"]["enable"] = True
        aug_par["photometric"]["params"] = self.config["gaussian_label"]["params"]
        augmentation = ImgAugTransform(**aug_par)
        # get label_2D
        # labels = points_to_2D(pnts, H, W)
        image = image[:, :, np.newaxis]
        heatmaps = augmentation(image)
        return heatmaps.squeeze()
