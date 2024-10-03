"""script for subpixel experiment (not tested)
"""

import logging
import torch
import torch.optim
import torch.utils.data

from loguru import logger
from ultrapoint.trainers.train_model_frontend import TrainModelFrontend


class TrainModelSubpixel(TrainModelFrontend):
    default_config = {
        "train_iter": 170000,
        "save_interval": 2000,
        "tensorboard_interval": 200,
        "model": {"subpixel": {"enable": False}},
    }

    def __init__(self, config, save_path, device=None):
        super().__init__(config, save_path, device)
        logger.info("Using: Train_model_subpixel")
        self.save_path = save_path
        self.cell_size = 8
        self.max_iter = config["train_iter"]
        self._train = True
        self._eval = True

    def train_val_sample(self, sample, n_iter=0, train=False):
        task = "train" if train else "val"
        tb_interval = self.config["tensorboard_interval"]

        losses, tb_imgs, tb_hist = {}, {}, {}
        ## get the inputs
        # logging.info('get input img and label')
        img, labels_2D, mask_2D = (
            sample["image"],
            sample["labels_2D"],
            sample["valid_mask"],
        )
        # img, labels = img.to(self.device), labels_2D.to(self.device)
        labels_res = sample["labels_res"]

        # variables
        batch_size, H, W = img.shape[0], img.shape[2], img.shape[3]
        self.batch_size = batch_size
        # print("batch_size: ", batch_size)
        Hc = H // self.cell_size
        Wc = W // self.cell_size

        # zero the parameter gradients
        self.optimizer.zero_grad()

        # extract patches
        # extract the patches from labels
        label_idx = labels_2D[...].nonzero()
        from src.ultrapoint.utils.losses import extract_patches

        patch_size = self.config["model"]["params"]["patch_size"]
        patches = extract_patches(
            label_idx.to(self.device), img.to(self.device), patch_size=patch_size
        )  # tensor [N, patch_size, patch_size]
        # patches = extract_patches(label_idx.to(device), labels_2D.to(device), patch_size=15) # tensor [N, patch_size, patch_size]
        # print("patches: ", patches.shape)

        patch_channels = self.config["model"]["params"].get("subpixel_channel", 1)
        if patch_channels == 2:
            patch_heat = extract_patches(
                label_idx.to(self.device), img.to(self.device), patch_size=patch_size
            )  # tensor [N, patch_size, patch_size]

        def label_to_points(labels_res, points):
            labels_res = labels_res.transpose(1, 2).transpose(2, 3).unsqueeze(1)
            points_res = labels_res[
                points[:, 0], points[:, 1], points[:, 2], points[:, 3], :
            ]  # tensor [N, 2]
            return points_res

        points_res = label_to_points(labels_res, label_idx)

        num_patches_max = 500
        # feed into the network
        pred_res = self.net(
            patches[:num_patches_max, ...].to(self.device)
        )  # tensor [1, N, 2]

        # loss function
        def get_loss(points_res, pred_res):
            loss = points_res - pred_res
            loss = torch.norm(loss, p=2, dim=-1).mean()
            return loss

        loss = get_loss(points_res[:num_patches_max, ...].to(self.device), pred_res)
        self.loss = loss

        losses.update({"loss": loss})
        tb_hist.update({"points_res_0": points_res[:, 0]})
        tb_hist.update({"points_res_1": points_res[:, 1]})
        tb_hist.update({"pred_res_0": pred_res[:, 0]})
        tb_hist.update({"pred_res_1": pred_res[:, 1]})
        tb_imgs.update({"patches": patches[:, ...].unsqueeze(1)})
        tb_imgs.update({"img": img})
        # forward + backward + optimize
        # if train:
        #     print("img: ", img.shape)
        #     outs, outs_warp = self.net(img.to(self.device)), self.net(img_warp.to(self.device), subpixel=self.subpixel)
        #     semi, coarse_desc = outs[0], outs[1]
        #     semi_warp, coarse_desc_warp = outs_warp[0], outs_warp[1]
        # else:
        #     with torch.no_grad():
        #         outs, outs_warp = self.net(img.to(self.device)), self.net(img_warp.to(self.device), subpixel=self.subpixel)
        #         semi, coarse_desc = outs[0], outs[1]
        #         semi_warp, coarse_desc_warp = outs_warp[0], outs_warp[1]
        #         pass

        # descriptor loss

        losses.update({"loss": loss})
        # print("losses: ", losses)

        if train:
            loss.backward()
            self.optimizer.step()

        self.tb_scalar_dict(losses, task)
        if n_iter % tb_interval == 0 or task == "val":
            logging.info(
                f"current iteration: {n_iter}, tensorboard_interval: {tb_interval}"
            )
            self.tb_images_dict(task, tb_imgs, max_img=5)
            self.tb_hist_dict(task, tb_hist)

        return loss.item()

    def tb_images_dict(self, task, tb_imgs, max_img=5):
        for element in list(tb_imgs):
            for idx in range(tb_imgs[element].shape[0]):
                if idx >= max_img:
                    break
                self.writer.add_image(
                    task + "-" + element + "/%d" % idx,
                    tb_imgs[element][idx, ...],
                    self.n_iter,
                )

    def tb_hist_dict(self, task, tb_dict):
        for element in list(tb_dict):
            self.writer.add_histogram(
                task + "-" + element, tb_dict[element], self.n_iter
            )
        pass
