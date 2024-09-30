import torch
import torch.optim
import torch.utils.data

from tqdm import tqdm
from loguru import logger
from utils.utils import flattenDetection
from models.model_wrap import SuperPointFrontend_torch


@torch.no_grad()
class Val_model_heatmap(SuperPointFrontend_torch):
    def __init__(self, config, device="cpu", verbose=False):
        self.config = config
        self.model = self.config["name"]
        self.params = self.config["params"]
        self.weights_path = self.config["pretrained"]
        self.device = device

        ## other parameters

        self.nms_dist = self.config["nms"]
        self.conf_thresh = self.config["detection_threshold"]
        self.nn_thresh = self.config[
            "nn_thresh"
        ]  # L2 descriptor distance for good match.
        self.cell = 8  # deprecated
        self.cell_size = 8  # Size of each output cell. Keep this fixed.
        self.border_remove = 4  # Remove points this close to the border.
        self.sparsemap = None
        self.heatmap = None  # np[batch, 1, H, W]
        self.pts = None
        self.pts_subpixel = None
        ## new variables
        self.pts_nms_batch = None
        self.desc_sparse_batch = None
        self.patches = None
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
        logger.info("successfully load pretrained model from: %s", self.weights_path)
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

    def run(self, images):
        """
        input:
            images: tensor[batch(1), 1, H, W]

        """
        from train_model_heatmap import TrainModelHeatmap
        from utils.var_dim import toNumpy

        train_agent = TrainModelHeatmap

        with torch.no_grad():
            outs = self.net(images)
        semi = outs["semi"]
        self.outs = outs

        channel = semi.shape[1]
        if channel == 64:
            heatmap = train_agent.flatten_64to1(semi, cell_size=self.cell_size)
        elif channel == 65:
            heatmap = flattenDetection(semi, tensor=True)

        return toNumpy(heatmap)

    def heatmap_to_pts(self, heatmap_np):
        pts_nms_batch = [self.getPtsFromHeatmap(h) for h in heatmap_np]  # [batch, H, W]
        self.pts_nms_batch = pts_nms_batch
        return pts_nms_batch

    def sparsify_descriptors(self):
        # pts_nms_batch = [self.getPtsFromHeatmap(h) for h in heatmap_np]
        desc_sparse_batch = [
            self.sample_desc_from_points(self.outs["desc"], pts)
            for pts in self.pts_nms_batch
        ]
        self.desc_sparse_batch = desc_sparse_batch
        return desc_sparse_batch


if __name__ == "__main__":
    filename = "assets/configs/magicpoint_repeatability_heatmap.yaml"
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

    # load frontend
    val_agent = Val_model_heatmap(config["model"], device=device)

    # take one sample
    for i, sample in tqdm(enumerate(test_loader)):
        if i > 1:
            break

        val_agent.loadModel()
        # points from heatmap
        img = sample["image"]
        print("image: ", img.shape)

        heatmap_batch = val_agent.run(img.to(device))  # heatmap: numpy [batch, 1, H, W]
        # heatmap to pts
        pts = val_agent.heatmap_to_pts()
        # print("pts: ", pts)
        print("pts[0]: ", pts[0].shape)
        print("pts: ", pts[0][:, :3])

        pts_subpixel = val_agent.soft_argmax_points(pts)
        print("subpixels: ", pts_subpixel[0][:, :3])

        # heatmap, pts to desc
        desc_sparse = val_agent.sparsify_descriptors()
        print("desc_sparse[0]: ", desc_sparse[0].shape)

# pts, desc, _, heatmap
