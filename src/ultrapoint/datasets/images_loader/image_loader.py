from loguru import logger
from pathlib import Path


class ImagesLoader:
    SUPPORTED_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".gif")

    @staticmethod
    def load_samples(images_path: str, labels_path: str = None):
        try:
            # Ensure the path exists
            base_path = Path(images_path)
            if not base_path.exists() or not base_path.is_dir():
                raise FileNotFoundError(
                    f"Image path {base_path} does not exist or is not a directory."
                )

            image_paths = [
                p
                for p in base_path.rglob("*")
                if p.suffix.lower() in ImagesLoader.SUPPORTED_EXTENSIONS
            ]
            if not image_paths:
                raise ValueError(
                    f"No images found in the specified directory: {base_path}"
                )

            names = [p.stem for p in image_paths]
            image_paths = [str(p) for p in image_paths]

            # Load labels if they are specified in the configuration
            if labels_path:
                labels_path = Path(labels_path)
                if not labels_path.exists() or not labels_path.is_dir():
                    raise FileNotFoundError(
                        f"Labels path {labels_path} does not exist or is not a directory."
                    )

                logger.info(f"Loading labels from: {labels_path}")
                samples = []
                for img, name in zip(image_paths, names):
                    label_file = labels_path / f"{name}.npz"
                    if label_file.exists():
                        samples.append(
                            {
                                "image": img,
                                "name": name,
                                "points": str(label_file),
                            }
                        )
                    else:
                        logger.warning(
                            f"Label file {label_file} not found for image {img}"
                        )
                return samples
            else:
                # Return images without labels
                return [
                    {"image": img, "name": name}
                    for img, name in zip(image_paths, names)
                ]

        except Exception as e:
            logger.error(f"Failed to load samples: {str(e)}")
            return []
