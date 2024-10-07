import numpy
import torch
import torchvision

from numpy.linalg import inv
from torch.utils.data import Dataset

from ultrapoint.datasets.image_loader import ImageLoader
from ultrapoint.utils.homographies import sample_homography
from ultrapoint.utils.image_helpers import read_image
from ultrapoint.utils.utils import compute_valid_mask
from ultrapoint.utils.photometric import (
    ImgAugTransform,
    imgPhotometric,
)
from ultrapoint.utils.utils import (
    inv_warp_image,
    inv_warp_image_batch,
)
from ultrapoint.datasets.data_tools import np_to_tensor
from ultrapoint.datasets.data_tools import warpLabels
from ultrapoint.utils.torch_helpers import squeeze_to_numpy


class ImagesDataset(Dataset):
    def __init__(
        self, transforms: torchvision.transforms.Compose, mode: str = "train", **config
    ):
        self._config = config
        self._transforms = transforms
        self._mode = mode
        self._samples = ImageLoader.load_samples(config)

        self._enable_photo_train = config["augmentation"]["photometric"]["enable_train"]
        self._enable_homo_train = config["augmentation"]["homographic"]["enable_train"]
        self._enable_homo_val = config["augmentation"]["homographic"]["enable_val"]
        self._enable_photo_val = config["augmentation"]["photometric"]["enable_val"]

        self._cell_size = 8
        self._resize = config.get("preprocessing", {}).get("resize", None)
        self._gaussian_label = config.get("gaussian_label", {}).get("enable", False)

    def __len__(self):
        return len(self._samples)

    def __getitem__(self, index):
        """
        :param index:
        :return:
            image: tensor (H, W, channel=1)
        """

        def points_to_2D(pnts, H, W):
            labels = numpy.zeros((H, W))
            pnts = pnts.astype(int)
            labels[pnts[:, 1], pnts[:, 0]] = 1
            return labels

        to_floatTensor = lambda x: torch.tensor(x).type(torch.FloatTensor)

        sample = self._samples[index]
        input = {}
        input.update(sample)
        img_o = read_image(sample["image"], self._resize, normalized=True)
        H, W = img_o.shape[0], img_o.shape[1]

        img_aug = img_o.copy()
        if (self._enable_photo_train == True and self._mode == "train") or (
            self._enable_photo_val and self._mode == "val"
        ):
            img_aug = imgPhotometric(
                img_o, self._config["augmentation"]
            )  # numpy array (H, W, 1)

        # img_aug = _preprocess(img_aug[:,:,numpy.newaxis])
        img_aug = torch.tensor(img_aug, dtype=torch.float32).view(-1, H, W)

        valid_mask = compute_valid_mask(
            torch.tensor([H, W]), inv_homography=torch.eye(3)
        )
        input.update({"image": img_aug})
        input.update({"valid_mask": valid_mask})

        if self._config["homography_adaptation"]["enable"]:
            # img_aug = torch.tensor(img_aug)
            homoAdapt_iter = self._config["homography_adaptation"]["num"]
            homographies = numpy.stack(
                [
                    sample_homography(
                        numpy.array([2, 2]),
                        shift=-1,
                        **self._config["homography_adaptation"]["homographies"][
                            "params"
                        ],
                    )
                    for i in range(homoAdapt_iter)
                ]
            )
            ##### use inverse from the sample homography
            homographies = numpy.stack([inv(homography) for homography in homographies])
            homographies[0, :, :] = numpy.identity(3)
            homographies = torch.tensor(homographies, dtype=torch.float32)
            inv_homographies = torch.stack(
                [torch.inverse(homographies[i, :, :]) for i in range(homoAdapt_iter)]
            )

            # images
            warped_img = inv_warp_image_batch(
                img_aug.squeeze().repeat(homoAdapt_iter, 1, 1, 1),
                inv_homographies,
                mode="bilinear",
            ).unsqueeze(0)

            # masks
            valid_mask = compute_valid_mask(
                torch.tensor([H, W]),
                inv_homography=inv_homographies,
                erosion_radius=self._config["augmentation"]["homographic"][
                    "valid_border_margin"
                ],
            )
            input.update(
                {
                    "image": warped_img.squeeze(),
                    "valid_mask": valid_mask,
                    "image_2D": img_aug,
                }
            )
            input.update(
                {"homographies": homographies, "inv_homographies": inv_homographies}
            )

        if self._config["labels"]:
            pnts = numpy.load(sample["points"])["pts"]
            labels = points_to_2D(pnts, H, W)
            labels_2D = to_floatTensor(labels[numpy.newaxis, :, :])
            input.update({"labels_2D": labels_2D})

            ## residual
            labels_res = torch.zeros((2, H, W)).type(torch.FloatTensor)
            input.update({"labels_res": labels_res})

            if (self._enable_homo_train == True and self._mode == "train") or (
                self._enable_homo_val and self._mode == "val"
            ):
                homography = sample_homography(
                    numpy.array([2, 2]),
                    shift=-1,
                    **self._config["augmentation"]["homographic"]["params"],
                )

                ##### use inverse from the sample homography
                homography = inv(homography)
                ######

                inv_homography = inv(homography)
                inv_homography = torch.tensor(inv_homography).to(torch.float32)
                homography = torch.tensor(homography).to(torch.float32)
                #                 img = torch.from_numpy(img)
                warped_img = inv_warp_image(
                    img_aug.squeeze(), inv_homography, mode="bilinear"
                ).unsqueeze(0)
                # warped_img = warped_img.squeeze().numpy()
                # warped_img = warped_img[:,:,np.newaxis]

                ##### check: add photometric #####

                # labels = torch.from_numpy(labels)
                # warped_labels = self.inv_warp_image(labels.squeeze(), inv_homography, mode='nearest').unsqueeze(0)
                ##### check #####
                warped_set = warpLabels(pnts, H, W, homography)
                warped_labels = warped_set["labels"]
                # if self.transform is not None:
                # warped_img = self.transform(warped_img)
                valid_mask = compute_valid_mask(
                    torch.tensor([H, W]),
                    inv_homography=inv_homography,
                    erosion_radius=self._config["augmentation"]["homographic"][
                        "valid_border_margin"
                    ],
                )

                input.update(
                    {
                        "image": warped_img,
                        "labels_2D": warped_labels,
                        "valid_mask": valid_mask,
                    }
                )

            if self._config["warped_pair"]["enable"]:
                homography = sample_homography(
                    numpy.array([2, 2]),
                    shift=-1,
                    **self._config["warped_pair"]["params"],
                )

                ##### use inverse from the sample homography
                homography = numpy.linalg.inv(homography)
                #####
                inv_homography = numpy.linalg.inv(homography)

                homography = torch.tensor(homography).type(torch.FloatTensor)
                inv_homography = torch.tensor(inv_homography).type(torch.FloatTensor)

                # photometric augmentation from original image

                # warp original image
                warped_img = torch.tensor(img_o, dtype=torch.float32)
                warped_img = inv_warp_image(
                    warped_img.squeeze(), inv_homography, mode="bilinear"
                ).unsqueeze(0)

                if (self._enable_photo_train == True and self._mode == "train") or (
                    self._enable_photo_val and self._mode == "val"
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
                if self._gaussian_label:
                    warped_labels_bi = warped_set["labels_bi"]
                    warped_labels_gaussian = self.gaussian_blur(
                        squeeze_to_numpy(warped_labels_bi)
                    )
                    warped_labels_gaussian = np_to_tensor(warped_labels_gaussian, H, W)
                    input["warped_labels_gaussian"] = warped_labels_gaussian
                    input.update({"warped_labels_bi": warped_labels_bi})

                input.update(
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
                    erosion_radius=self._config["warped_pair"]["valid_border_margin"],
                )  # can set to other value
                input.update({"warped_valid_mask": valid_mask})
                input.update(
                    {"homographies": homography, "inv_homographies": inv_homography}
                )

            if self._gaussian_label:
                labels_gaussian = self.gaussian_blur(squeeze_to_numpy(labels_2D))
                labels_gaussian = np_to_tensor(labels_gaussian, H, W)
                input["labels_2D_gaussian"] = labels_gaussian

        name = sample["name"]
        input.update({"name": name, "scene_name": "./"})  # dummy scene name

        return input

    ## util functions
    def gaussian_blur(self, image: numpy.ndarray) -> numpy.ndarray:
        """
        image: np [H, W]
        return:
            blurred_image: np [H, W]
        """
        augmentation_config = {
            "photometric": {
                "enable": True,
                "params": self._config["gaussian_label"]["params"],
            }
        }
        augmentation = ImgAugTransform(**augmentation_config)
        image = image[:, :, numpy.newaxis]
        heatmaps = augmentation(image)
        return heatmaps.squeeze()
