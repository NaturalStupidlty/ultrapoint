import cv2
import numpy as np

from numpy.random import randint
from imgaug import augmenters


class ImageAugmentation:
    def __init__(self, **config):
        self.augmentations = augmenters.Sequential(
            [
                augmenters.Sometimes(0.25, augmenters.GaussianBlur(sigma=(0, 3.0))),
                augmenters.Sometimes(
                    0.25,
                    augmenters.OneOf(
                        [
                            augmenters.Dropout(p=(0, 0.1)),
                            augmenters.CoarseDropout(0.1, size_percent=0.5),
                        ]
                    ),
                ),
                augmenters.Sometimes(
                    0.25,
                    augmenters.AdditiveGaussianNoise(
                        loc=0, scale=(0.0, 0.05), per_channel=0.5
                    ),
                ),
            ]
        )

        if config["photometric"]["enable"]:
            params = config["photometric"]["params"]
            aug_all = []
            if params.get("random_brightness", False):
                change = params["random_brightness"]["max_abs_change"]
                aug = augmenters.Add((-change, change))
                aug_all.append(aug)
            if params.get("random_contrast", False):
                change = params["random_contrast"]["strength_range"]
                aug = augmenters.LinearContrast((change[0], change[1]))
                aug_all.append(aug)
            if params.get("additive_gaussian_noise", False):
                change = params["additive_gaussian_noise"]["stddev_range"]
                aug = augmenters.AdditiveGaussianNoise(scale=(change[0], change[1]))
                aug_all.append(aug)
            if params.get("additive_speckle_noise", False):
                change = params["additive_speckle_noise"]["prob_range"]
                aug = augmenters.ImpulseNoise(p=(change[0], change[1]))
                aug_all.append(aug)
            if params.get("motion_blur", False):
                change = params["motion_blur"]["max_kernel_size"]
                change = randint(3, change) if change > 3 else change
                aug = augmenters.Sometimes(0.5, augmenters.MotionBlur(change))
                aug_all.append(aug)
            if params.get("gaussian_blur", False):
                change = params["gaussian_blur"]["sigma"]
                aug = augmenters.GaussianBlur(sigma=change)
                aug_all.append(aug)

            self.augmentations = augmenters.Sequential(aug_all)

        else:
            self.augmentations = augmenters.Sequential(
                [
                    augmenters.Noop(),
                ]
            )

    def __call__(self, img):
        img = np.array(img)
        img = (img * 255).astype(np.uint8)
        img = self.augmentations.augment_image(img)
        img = img.astype(np.float32) / 255
        return img


class customizedTransform:

    def additive_shade(
        self,
        image,
        nb_ellipses=20,
        transparency_range=[-0.5, 0.8],
        kernel_size_range=[250, 350],
    ):
        def _py_additive_shade(img):
            min_dim = min(img.shape[:2]) / 4
            mask = np.zeros(img.shape[:2], np.uint8)
            for i in range(nb_ellipses):
                ax = int(max(np.random.rand() * min_dim, min_dim / 5))
                ay = int(max(np.random.rand() * min_dim, min_dim / 5))
                max_rad = max(ax, ay)
                x = np.random.randint(max_rad, img.shape[1] - max_rad)  # center
                y = np.random.randint(max_rad, img.shape[0] - max_rad)
                angle = np.random.rand() * 90
                cv2.ellipse(mask, (x, y), (ax, ay), angle, 0, 360, 255, -1)

            transparency = np.random.uniform(*transparency_range)
            kernel_size = np.random.randint(*kernel_size_range)
            if (kernel_size % 2) == 0:  # kernel_size has to be odd
                kernel_size += 1
            mask = cv2.GaussianBlur(
                mask.astype(np.float32), (kernel_size, kernel_size), 0
            )
            #             shaded = img * (1 - transparency * mask[..., np.newaxis] / 255.)
            shaded = img * (1 - transparency * mask[..., np.newaxis] / 255.0)
            return np.clip(shaded, 0, 255)

        shaded = _py_additive_shade(image)
        return shaded

    def __call__(self, img, **config):
        if config["photometric"]["params"]["additive_shade"]:
            params = config["photometric"]["params"]
            img = self.additive_shade(img * 255, **params["additive_shade"])
        return img / 255


def imgPhotometric(img, augmentations_config: dict):
    """
    :param img:
        numpy (H, W)
    :return:
    """
    augmentation = ImageAugmentation(**augmentations_config)
    img = img[:, :, np.newaxis]
    img = augmentation(img)
    cusAug = customizedTransform()
    img = cusAug(img, **augmentations_config)
    return img
