import random
import albumentations
import numpy as np


class ImageAugmentation:
    def __init__(self, **config):
        self._augmentations = []
        if config["photometric"]["enable"]:
            for augmentation in config["photometric"]["augmentations"]:
                if not augmentation:
                    continue
                name, parameters = augmentation.popitem()
                self._augmentations.append(getattr(albumentations, name)(**parameters))

            if config["photometric"]["random_order"]:
                random.shuffle(self._augmentations)

        self._augmentations = albumentations.Compose(self._augmentations)

    @property
    def augmentations(self) -> albumentations.Compose:
        return self._augmentations

    @augmentations.setter
    def augmentations(self, value: albumentations.Compose) -> None:
        self._augmentations = value

    def __call__(
        self,
        image: np.ndarray,
        normalize: bool = False,
        grayscale: bool = False,
        **config
    ):
        # TODO: ???
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)

        if len(image.shape) == 2:
            grayscale = True
            image = image[:, :, np.newaxis]

        if image.shape[-1] == 1:
            grayscale = True
            image = np.concatenate([image] * 3, axis=-1)

        image = (image * 255).astype(np.uint8)
        augmented = self._augmentations(image=image)
        image = augmented["image"].astype(np.float32)

        if normalize:
            image /= 255

        if grayscale:
            image = np.mean(image, axis=-1, keepdims=True)

        return image
