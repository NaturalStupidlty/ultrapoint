import numpy as np
import torch.utils.data as data

from PIL import Image
from pathlib import Path
from typing import Tuple

from ultrapoint.utils.image_helpers import read_image


class HPatchesDataset(data.Dataset):
    def __init__(self, mode: str = "val", **config):
        self._config = config
        self._mode = mode
        self._resize = config["preprocessing"]["resize"]
        self._alteration = config.get("alteration", "all")
        self._truncate = config.get("truncate", None)
        self._samples = {}

        self._init_dataset()

    def __getitem__(self, index):
        image = read_image(self._samples["image"][index], self._resize, normalized=True)
        warped_image = read_image(self._samples["warped_image"][index])
        homography = self.rescale_homography(
            Image.open(self._samples["image"][index]).size,
            self._samples["homography"][index],
        )

        return {
            "image": np.expand_dims(image, axis=0),
            "warped_image": np.expand_dims(warped_image, axis=0),
            "homography": homography,
        }

    def __len__(self):
        return len(self._samples["image"])

    def rescale_homography(
        self, image_size: Tuple[int, int], homography_matrix: np.ndarray
    ):
        scale_factor = max(
            self._resize[0] / image_size[0], self._resize[1] / image_size[1]
        )
        transformation_matrix = np.array(
            [
                [1, 1, scale_factor],
                [1, 1, scale_factor],
                [1 / scale_factor, 1 / scale_factor, 1],
            ]
        )

        return homography_matrix * transformation_matrix

    def _init_dataset(self):
        dataset_path = Path(self._config["path"])
        folder_paths = [x for x in dataset_path.iterdir() if x.is_dir()]

        image_paths = []
        warped_image_paths = []
        homographies = []

        for path in folder_paths:
            if self._alteration == "i" and path.stem[0] != "i":
                continue
            if self._alteration == "v" and path.stem[0] != "v":
                continue

            file_ext = ".ppm"
            num_images = 5

            for i in range(2, 2 + num_images):
                image_paths.append(str(Path(path, "1" + file_ext)))
                warped_image_paths.append(str(Path(path, str(i) + file_ext)))
                homographies.append(np.loadtxt(str(Path(path, "H_1_" + str(i)))))

        if self._truncate:
            image_paths = image_paths[: self._truncate]
            warped_image_paths = warped_image_paths[: self._truncate]
            homographies = homographies[: self._truncate]

        self._samples = {
            "image": image_paths,
            "warped_image": warped_image_paths,
            "homography": homographies,
        }
