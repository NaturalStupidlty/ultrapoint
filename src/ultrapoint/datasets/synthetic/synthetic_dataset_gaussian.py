import cv2
import shutil
import tarfile
import torch.utils.data as data
import torch
import numpy

from typing import Union, List
from imageio import imread
from tqdm import tqdm
from loguru import logger
from pathlib import Path

from ultrapoint.datasets.synthetic import synthetic_dataset
from src.ultrapoint.utils.homographies import (
    sample_homography as sample_homography,
)
from ultrapoint.utils.utils import compute_mask
from ultrapoint.utils.utils import inv_warp_image, warp_points
from ultrapoint.utils.utils import filter_points
from ultrapoint.datasets.augmentations import ImageAugmentation


def load_as_float(path):
    return imread(path).astype(numpy.float32) / 255


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
        mode="train",
        **config,
    ):
        self._config = config
        self._data_path = config.get("path", "/tmp")
        self._shuffle = config.get("shuffle", False)
        self._mode = mode
        self._config["augmentation"]["photometric"]["enable"] = config["augmentation"][
            "photometric"
        ][f"enable_train" if self._mode == "train" else "enable_val"]
        self._use_homographic_augmentation = (
                (self._mode == "train" and self._config["augmentation"]["homographic"]["enable_train"])
                or
                (self._mode != "train" and self._config["augmentation"]["homographic"]["enable_val"])
        )
        self._use_photometric_augmentation = (
                (self._mode == "train" and self._config["augmentation"]["photometric"]["enable_train"])
                or
                (self._mode != "train" and self._config["augmentation"]["photometric"]["enable_val"])
        )

        self._augmentation = ImageAugmentation(**self._config["augmentation"])

        # Parse drawing primitives
        primitives = self.setup_primitives(config["primitives"])
        logger.info(primitives)

        basepath = Path(self._data_path)
        basepath.mkdir(parents=True, exist_ok=True)

        splits = {s: {"images": [], "points": []} for s in [self._mode]}
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
            if config["truncate"] is not None:
                truncate = config["truncate"].get(primitive, 1)
            else:
                truncate = 1
            path = Path(temp_dir, primitive)
            for s in splits:
                e = [str(p) for p in Path(path, "images", s).iterdir()]
                f = [p.replace("images", "points") for p in e]
                f = [p.replace(".png", ".npy") for p in f]
                splits[s]["images"].extend(e[: int(truncate * len(e))])
                splits[s]["points"].extend(f[: int(truncate * len(f))])

        # Shuffle
        for s in splits:
            perm = numpy.random.RandomState(0).permutation(len(splits[s]["images"]))
            for obj in ["images", "points"]:
                splits[s][obj] = numpy.array(splits[s][obj])[perm].tolist()

        self.crawl_folders(splits)
        if self._shuffle:
            numpy.random.shuffle(self.samples)

    def __getitem__(self, index):
        """
        :param index:
        :return:
            labels_2D: tensor(1, H, W)
            image: tensor(1, H, W)
        """

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

        sample = self.samples[index]
        img = load_as_float(sample["image"])
        H, W = img.shape[0], img.shape[1]
        self.H = H
        self.W = W
        pnts = numpy.load(sample["points"])  # (y, x)
        pnts = torch.tensor(pnts).float()
        pnts = torch.stack((pnts[:, 1], pnts[:, 0]), dim=1)  # (x, y)
        pnts = filter_points(pnts, torch.tensor([W, H]))
        sample = {}

        # print('pnts: ', pnts[:5])
        # print('--1', pnts)
        labels_2D = get_labels(pnts, H, W)
        sample.update({"labels_2D": labels_2D.unsqueeze(0)})

        if self._use_photometric_augmentation:
            from src.ultrapoint.utils.utils import (
                homography_scaling_torch as homography_scaling,
            )
            from numpy.linalg import inv

            homography = sample_homography(
                numpy.array([2, 2]),
                shift=-1,
                **self._config["augmentation"]["homographic"]["params"],
            )

            ##### use inverse from the sample homography
            homography = inv(homography)
            ######

            homography = torch.tensor(homography).float()
            inv_homography = homography.inverse()
            img = torch.from_numpy(img)
            warped_img = inv_warp_image(img.squeeze(), inv_homography, mode="bilinear")
            warped_img = warped_img.squeeze().numpy()
            warped_img = warped_img[:, :, numpy.newaxis]

            warped_pnts = warp_points(pnts, homography_scaling(homography, H, W))
            warped_pnts = filter_points(warped_pnts, torch.tensor([W, H]))

            warped_img = torch.tensor(warped_img, dtype=torch.float32).view(-1, H, W)
            sample["image"] = warped_img

            mask = compute_mask(
                torch.tensor([H, W]),
                inv_homography=inv_homography,
                erosion_radius=self._config["augmentation"]["homographic"][
                    "valid_border_margin"
                ],
            )  # can set to other value
            sample.update({"mask": mask})

            labels_2D = get_labels(warped_pnts, H, W)
            sample.update({"labels_2D": labels_2D.unsqueeze(0)})

            labels_res = get_label_res(H, W, warped_pnts)
        else:
            img = img[:, :, numpy.newaxis]
            img = torch.tensor(img, dtype=torch.float32).view(-1, H, W)
            sample["image"] = img
            # sample = {'image': img, 'labels_2D': labels}
            mask = compute_mask(torch.tensor([H, W]), inv_homography=torch.eye(3))
            sample.update({"mask": mask})
            labels_res = get_label_res(H, W, pnts)

        if self._use_photometric_augmentation:
            sample["image"] = torch.tensor(self._augmentation(sample["image"].squeeze().numpy()), dtype=torch.float32).view(-1, H, W)

        sample.update({"labels_res": labels_res})

        if self._config["warped_pair"]["enable"]:
            from src.ultrapoint.datasets.data_tools import warpLabels

            homography = sample_homography(
                numpy.array([2, 2]), shift=-1, **self._config["warped_pair"]["params"]
            )

            ##### use inverse from the sample homography
            homography = numpy.linalg.inv(homography)
            #####
            inv_homography = numpy.linalg.inv(homography)

            homography = torch.tensor(homography).type(torch.FloatTensor)
            inv_homography = torch.tensor(inv_homography).type(torch.FloatTensor)

            # photometric augmentation from original image

            # warp original image
            warped_img = img.type(torch.FloatTensor)
            warped_img = inv_warp_image(
                warped_img.squeeze(), inv_homography, mode="bilinear"
            ).unsqueeze(0)
            warped_img = warped_img.view(-1, H, W)

            # warped_labels = warpLabels(pnts, H, W, homography)
            warped_set = warpLabels(pnts, H, W, homography, bilinear=True)
            warped_labels = warped_set["labels"]
            warped_res = warped_set["res"]
            warped_res = warped_res.transpose(1, 2).transpose(0, 1)

            sample.update(
                {
                    "warped_img": warped_img,
                    "warped_labels": warped_labels,
                    "warped_res": warped_res,
                }
            )

            # print('erosion_radius', self.config['warped_pair']['valid_border_margin'])
            mask = compute_mask(
                torch.tensor([H, W]),
                inv_homography=inv_homography,
                erosion_radius=self._config["warped_pair"]["valid_border_margin"],
            )  # can set to other value
            sample.update({"warped_mask": mask})
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
            numpy.random.RandomState(config["generation"]["random_seed"])
        )
        for split, size in self._config["generation"]["split_sizes"].items():
            im_dir, pts_dir = [Path(temp_dir, i, split) for i in ["images", "points"]]
            im_dir.mkdir(parents=True, exist_ok=True)
            pts_dir.mkdir(parents=True, exist_ok=True)

            description = f"Generating {primitive} {split} set"
            for i in tqdm(range(size), desc=description, leave=False):
                image = synthetic_dataset.generate_background(
                    config["generation"]["image_size"],
                    **config["generation"]["params"]["generate_background"],
                )
                points = numpy.array(
                    getattr(synthetic_dataset, primitive)(
                        image, **config["generation"]["params"].get(primitive, {})
                    )
                )
                points = numpy.flip(points, 1)  # reverse convention with opencv
                points = (
                    points
                    * numpy.array(config["preprocessing"]["resize"], float)
                    / numpy.array(config["generation"]["image_size"], float)
                )
                image = cv2.resize(
                    image,
                    tuple(config["preprocessing"]["resize"][::-1]),
                    interpolation=cv2.INTER_LINEAR,
                )

                cv2.imwrite(str(Path(im_dir, "{}.png".format(i))), image)
                numpy.save(Path(pts_dir, "{}.npy".format(i)), points)

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
            splits[self._mode]["images"], splits[self._mode]["points"]
        ):
            sample = {"image": img, "points": pnts}
            sequence_set.append(sample)
        self.samples = sequence_set
