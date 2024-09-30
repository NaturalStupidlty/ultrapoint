"""script for subpixel experiment (not tested)
"""

import torch
import torch.optim
import torch.utils.data
from tqdm import tqdm
from loguru import logger


@torch.no_grad()
class Val_model_subpixel(object):
    def __init__(self, config, device="cpu", verbose=False):
        self.config = config
        self.model = self.config["name"]
        self.params = self.config["params"]
        self.weights_path = self.config["pretrained"]
        self.device = device
        pass

    def loadModel(self):
        # model = 'SuperPointNet'
        # params = self.config['model']['subpixel']['params']
        from utils.loader import modelLoader

        self.net = modelLoader(model=self.model, **self.params)

        checkpoint = torch.load(
            self.weights_path, map_location=lambda storage, loc: storage
        )
        self.net.load_state_dict(checkpoint["model_state_dict"])

        self.net = self.net.to(self.device)
        logger.info(f"Successfully load pretrained model from: {self.weights_path}")
        pass

    def extract_patches(self, label_idx, img):
        """
        input:
            label_idx: tensor [N, 4]: (batch, 0, y, x)
            img: tensor [batch, channel(1), H, W]
        """
        from utils.losses import extract_patches

        patch_size = self.config["params"]["patch_size"]
        patches = extract_patches(
            label_idx.to(self.device), img.to(self.device), patch_size=patch_size
        )
        return patches
        pass

    def run(self, patches):
        """ """
        with torch.no_grad():
            pred_res = self.net(patches)
        return pred_res
        pass


if __name__ == "__main__":
    # filename = 'configs/magicpoint_shapes_subpix.yaml'
    filename = "assets/configs/magicpoint_repeatability.yaml"
    import yaml

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_default_dtype(torch.float32)

    with open(filename, "r") as f:
        config = yaml.safe_load(f)

    task = config["data"]["dataset"]
    # data loading
    from utils.loader import DataLoaderTest as dataLoader

    data = dataLoader(config, dataset="hpatches")
    test_set, test_loader = data["test_set"], data["test_loader"]

    # take one sample
    for i, sample in tqdm(enumerate(test_loader)):
        if i > 1:
            break

        val_agent = Val_model_subpixel(config["subpixel"], device=device)
        val_agent.loadModel()
        # points from heatmap
        img = sample["image"]
        print("image: ", img.shape)
        points = torch.tensor([[1, 2], [3, 4]])

        def points_to_4d(points):
            num_of_points = points.shape[0]
            cols = torch.zeros(num_of_points, 1).float()
            points = torch.cat((cols, cols, points.float()), dim=1)
            return points

        label_idx = points_to_4d(points)
        # concat points to be (batch, 0, y, x)
        patches = val_agent.extract_patches(label_idx, img)
        points_res = val_agent.run(patches)
