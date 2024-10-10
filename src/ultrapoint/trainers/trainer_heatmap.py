import numpy
import torch
import torch.optim
import torch.nn as nn
import torch.utils.data

from loguru import logger
from ultrapoint.utils.utils import flattenDetection
from ultrapoint.utils.utils import precisionRecall_torch
from ultrapoint.trainers.trainer import Trainer
from ultrapoint.loggers.loguru import log_losses


class TrainerHeatmap(Trainer):
    """
    heatmap: torch (batch_size, H, W, 1)
    dense_desc: torch (batch_size, H, W, 256)
    pts: [batch_size, np (N, 3)]
    desc: [batch_size, np(256, N)]
    """

    def __init__(self, config, save_path, device=None):
        super().__init__(config, save_path, device)

    def detector_loss(self, input, target, mask=None, loss_type="softmax"):
        """
        # apply loss on detectors, default is softmax
        :param input: prediction
            tensor [batch_size, 65, Hc, Wc]
        :param target: constructed from labels
            tensor [batch_size, 65, Hc, Wc]
        :param mask: valid region in an image
            tensor [batch_size, 1, Hc, Wc]
        :param loss_type:
            str (l2 or softmax)
            softmax is used in original paper
        :return: normalized loss
            tensor
        """
        if loss_type == "l2":
            loss_func = nn.MSELoss(reduction="mean")
            loss = loss_func(input, target)
        elif loss_type == "softmax":
            loss_func_BCE = nn.BCELoss(reduction="none").cuda()
            loss = loss_func_BCE(nn.functional.softmax(input, dim=1), target)
            loss = (loss.sum(dim=1) * mask).sum()
            loss = loss / (mask.sum() + 1e-10)
        return loss

    def process_sample(self, sample, iteration=0, task="val"):
        """
        # key function
        :param sample:
        :param iteration:
        :param train:
        :return:
        """
        assert task in ["train", "val"], "task should be either train or val"

        to_floatTensor = lambda x: torch.tensor(x).type(torch.FloatTensor)

        tb_interval = self._config["tensorboard_interval"]
        if_warp = self._config["data"]["warped_pair"]["enable"]

        self.scalar_dict, self.images_dict, self.hist_dict = {}, {}, {}
        ## get the inputs
        # logging.info('get input img and label')
        img, labels_2D, mask_2D = (
            sample["image"],
            sample["labels_2D"],
            sample["valid_mask"],
        )
        # img, labels = img.to(self.device), labels_2D.to(self.device)

        # variables
        batch_size, H, W = img.shape[0], img.shape[2], img.shape[3]
        self.batch_size = batch_size
        det_loss_type = self._config["model"]["detector_loss"]["loss_type"]
        # print("batch_size: ", batch_size)
        Hc = H // self._cell_size
        Wc = W // self._cell_size

        # warped images
        # img_warp, labels_warp_2D, mask_warp_2D = sample['warped_img'].to(self.device), \
        #     sample['warped_labels'].to(self.device), \
        #     sample['warped_valid_mask'].to(self.device)
        if if_warp:
            img_warp, labels_warp_2D, mask_warp_2D = (
                sample["warped_img"],
                sample["warped_labels"],
                sample["warped_valid_mask"],
            )

        # homographies
        # mat_H, mat_H_inv = \
        # sample['homographies'].to(self.device), sample['inv_homographies'].to(self.device)
        if if_warp:
            mat_H, mat_H_inv = sample["homographies"], sample["inv_homographies"]

        # zero the parameter gradients
        self.optimizer.zero_grad()

        # forward + backward + optimize
        if task == "train":
            # print("img: ", img.shape, ", img_warp: ", img_warp.shape)
            outs = self.net(img.to(self._device))
            semi, coarse_desc = outs["semi"], outs["desc"]
            if if_warp:
                outs_warp = self.net(img_warp.to(self._device))
                semi_warp, coarse_desc_warp = outs_warp["semi"], outs_warp["desc"]
        else:
            with torch.no_grad():
                outs = self.net(img.to(self._device))
                semi, coarse_desc = outs["semi"], outs["desc"]
                if if_warp:
                    outs_warp = self.net(img_warp.to(self._device))
                    semi_warp, coarse_desc_warp = outs_warp["semi"], outs_warp["desc"]
                pass

        # detector loss
        from src.ultrapoint.utils.utils import labels2Dto3D

        labels_2D = sample["labels_2D"]
        if if_warp:
            warped_labels = sample["warped_labels"]

        add_dustbin = False
        if det_loss_type == "l2":
            add_dustbin = False
        elif det_loss_type == "softmax":
            add_dustbin = True

        labels_3D = labels2Dto3D(
            labels_2D.to(self._device),
            cell_size=self._cell_size,
            add_dustbin=add_dustbin,
        ).float()
        mask_3D_flattened = self.get_masks(
            mask_2D, self._cell_size, device=self._device
        )
        loss_det = self.detector_loss(
            input=outs["semi"],
            target=labels_3D.to(self._device),
            mask=mask_3D_flattened.to(self._device),
            loss_type=det_loss_type,
        )
        # warp
        if if_warp:
            labels_3D = labels2Dto3D(
                warped_labels.to(self._device),
                cell_size=self._cell_size,
                add_dustbin=add_dustbin,
            ).float()
            mask_3D_flattened = self.get_masks(
                mask_warp_2D, self._cell_size, device=self._device
            )
            loss_det_warp = self.detector_loss(
                input=outs_warp["semi"],
                target=labels_3D.to(self._device),
                mask=mask_3D_flattened.to(self._device),
                loss_type=det_loss_type,
            )
        else:
            loss_det_warp = torch.tensor([0]).float().to(self._device)

        ## get labels, masks, loss for detection
        # labels3D_in_loss = self.getLabels(labels_2D, self._cell_size, device=self.device)
        # mask_3D_flattened = self.getMasks(mask_2D, self._cell_size, device=self.device)
        # loss_det = self.get_loss(semi, labels3D_in_loss, mask_3D_flattened, device=self.device)

        ## warping
        # labels3D_in_loss = self.getLabels(labels_warp_2D, self._cell_size, device=self.device)
        # mask_3D_flattened = self.getMasks(mask_warp_2D, self._cell_size, device=self.device)
        # loss_det_warp = self.get_loss(semi_warp, labels3D_in_loss, mask_3D_flattened, device=self.device)

        mask_desc = mask_3D_flattened.unsqueeze(1)
        lambda_loss = self._config["model"]["lambda_loss"]
        # print("mask_desc: ", mask_desc.shape)
        # print("mask_warp_2D: ", mask_warp_2D.shape)

        # descriptor loss
        if lambda_loss > 0:
            assert if_warp == True, "need a pair of images"
            loss_desc, mask, positive_dist, negative_dist = self._descriptor_loss(
                coarse_desc,
                coarse_desc_warp,
                mat_H,
                mask_valid=mask_desc,
                device=self._device,
                **self._desc_params,
            )
        else:
            ze = torch.tensor([0]).to(self._device)
            loss_desc, positive_dist, negative_dist = ze, ze, ze

        loss = loss_det + loss_det_warp
        if lambda_loss > 0:
            loss += lambda_loss * loss_desc

        ##### try to minimize the error ######
        add_res_loss = False
        if add_res_loss and iteration % 10 == 0:
            heatmap_org = flattenDetection(semi)
            heatmap_org_nms_batch = self.heatmap_to_nms(
                self.images_dict, heatmap_org, name="heatmap_org"
            )
            if if_warp:
                heatmap_warp = flattenDetection(semi_warp)
                heatmap_warp_nms_batch = self.heatmap_to_nms(
                    self.images_dict, heatmap_warp, name="heatmap_warp"
                )

            # original: pred
            ## check the loss on given labels!
            outs_res = self.get_residual_loss(
                sample["labels_2D"]
                * to_floatTensor(heatmap_org_nms_batch).unsqueeze(1),
                heatmap_org,
                sample["labels_res"],
            )
            loss_res_ori = (outs_res["loss"] ** 2).mean()
            # warped: pred
            if if_warp:
                outs_res_warp = self.get_residual_loss(
                    sample["warped_labels"]
                    * to_floatTensor(heatmap_warp_nms_batch).unsqueeze(1),
                    heatmap_warp,
                    sample["warped_res"],
                )
                loss_res_warp = (outs_res_warp["loss"] ** 2).mean()
            else:
                loss_res_warp = torch.tensor([0]).to(self._device)
            loss_res = loss_res_ori + loss_res_warp
            # print("loss_res requires_grad: ", loss_res.requires_grad)
            loss += loss_res
            self.scalar_dict.update(
                {"loss_res_ori": loss_res_ori, "loss_res_warp": loss_res_warp}
            )

        #######################################

        self.loss = loss

        self.scalar_dict.update(
            {
                "loss": loss,
                "loss_det": loss_det,
                "loss_det_warp": loss_det_warp,
                "positive_dist": positive_dist,
                "negative_dist": negative_dist,
            }
        )

        self.input_to_imgDict(sample, self.images_dict)

        if task == "train":
            loss.backward()
            self.optimizer.step()

        if iteration % tb_interval == 0 or task == "val":
            logger.info(
                f"current iteration: {iteration}, tensorboard_interval: {tb_interval}",
            )

            # add clean map to tensorboard
            ## semi_warp: flatten, to_numpy

            heatmap_org = flattenDetection(semi)
            heatmap_org_nms_batch = self.heatmap_to_nms(
                self.images_dict, heatmap_org, name="heatmap_org"
            )
            if if_warp:
                heatmap_warp = flattenDetection(semi_warp)
                heatmap_warp_nms_batch = self.heatmap_to_nms(
                    self.images_dict, heatmap_warp, name="heatmap_warp"
                )

            def update_overlap(
                images_dict, labels_warp_2D, heatmap_nms_batch, img_warp, name
            ):
                # image overlap
                from src.ultrapoint.utils.draw import img_overlap

                # overlap label, nms, img
                nms_overlap = [
                    img_overlap(
                        torch_to_numpy(labels_warp_2D[i]),
                        heatmap_nms_batch[i],
                        torch_to_numpy(img_warp[i]),
                    )
                    for i in range(heatmap_nms_batch.shape[0])
                ]
                nms_overlap = numpy.stack(nms_overlap, axis=0)
                images_dict.update({name + "_nms_overlap": nms_overlap})

            from src.ultrapoint.utils.torch_helpers import torch_to_numpy

            update_overlap(
                self.images_dict,
                labels_2D,
                heatmap_org_nms_batch[numpy.newaxis, ...],
                img,
                "original",
            )

            update_overlap(
                self.images_dict,
                labels_2D,
                torch_to_numpy(heatmap_org),
                img,
                "original_heatmap",
            )
            if if_warp:
                update_overlap(
                    self.images_dict,
                    labels_warp_2D,
                    heatmap_warp_nms_batch[numpy.newaxis, ...],
                    img_warp,
                    "warped",
                )
                update_overlap(
                    self.images_dict,
                    labels_warp_2D,
                    torch_to_numpy(heatmap_warp),
                    img_warp,
                    "warped_heatmap",
                )
            # residuals
            from src.ultrapoint.utils.losses import do_log

            pr_mean = self.batch_precision_recall(
                to_floatTensor(heatmap_org_nms_batch[:, numpy.newaxis, ...]),
                sample["labels_2D"],
            )
            self.scalar_dict.update(pr_mean)

            log_losses(self.scalar_dict, task)
            self._tensorboard_logger.tb_images_dict(
                iteration, task, self.images_dict, max_img=2
            )
            self._tensorboard_logger.tb_hist_dict(iteration, task, self.hist_dict)

        self._tensorboard_logger.tb_scalar_dict(iteration, self.scalar_dict, task)

        return loss.item()

    def heatmap_to_nms(self, images_dict, heatmap, name):
        """
        return:
            heatmap_nms_batch: np [batch, H, W]
        """
        from src.ultrapoint.utils.torch_helpers import torch_to_numpy

        heatmap_np = torch_to_numpy(heatmap)
        heatmap_nms_batch = [self.heatmap_nms(h) for h in heatmap_np]  # [batch, H, W]
        heatmap_nms_batch = numpy.stack(heatmap_nms_batch, axis=0)
        images_dict.update(
            {name + "_nms_batch": heatmap_nms_batch[:, numpy.newaxis, ...]}
        )
        return heatmap_nms_batch

    def get_residual_loss(self, labels_2D, heatmap, labels_res):
        if abs(labels_2D).sum() == 0:
            return
        outs_res = self.pred_soft_argmax(
            labels_2D, heatmap, labels_res, patch_size=5, device=self._device
        )
        return outs_res

    @staticmethod
    def batch_precision_recall(batch_pred, batch_labels):
        precision_recall_list = []
        for i in range(batch_labels.shape[0]):
            precision_recall = precisionRecall_torch(batch_pred[i], batch_labels[i])
            precision_recall_list.append(precision_recall)
        precision = numpy.mean(
            [
                precision_recall["precision"]
                for precision_recall in precision_recall_list
            ]
        )
        recall = numpy.mean(
            [precision_recall["recall"] for precision_recall in precision_recall_list]
        )
        return {"precision": precision, "recall": recall}

    @staticmethod
    def pred_soft_argmax(labels_2D, heatmap, labels_res, patch_size=5, device="cuda"):
        """

        return:
            dict {'loss': mean of difference btw pred and res}
        """
        from src.ultrapoint.utils.losses import norm_patches

        outs = {}
        # extract patches
        from src.ultrapoint.utils.losses import extract_patches
        from src.ultrapoint.utils.losses import soft_argmax_2d

        label_idx = labels_2D[...].nonzero().long()

        # patch_size = self.config['params']['patch_size']
        patches = extract_patches(
            label_idx.to(device), heatmap.to(device), patch_size=patch_size
        )
        # norm patches
        patches = norm_patches(patches)

        # predict offsets
        from src.ultrapoint.utils.losses import do_log

        patches_log = do_log(patches)
        # soft_argmax
        dxdy = soft_argmax_2d(
            patches_log, normalized_coordinates=False
        )  # tensor [B, N, patch, patch]
        dxdy = dxdy.squeeze(1)  # tensor [N, 2]
        dxdy = dxdy - patch_size // 2

        # extract residual
        def ext_from_points(labels_res, points):
            """
            input:
                labels_res: tensor [batch, channel, H, W]
                points: tensor [N, 4(pos0(batch), pos1(0), pos2(H), pos3(W) )]
            return:
                tensor [N, channel]
            """
            labels_res = labels_res.transpose(1, 2).transpose(2, 3).unsqueeze(1)
            points_res = labels_res[
                points[:, 0], points[:, 1], points[:, 2], points[:, 3], :
            ]  # tensor [N, 2]
            return points_res

        points_res = ext_from_points(labels_res, label_idx)

        # loss
        outs["pred"] = dxdy
        outs["points_res"] = points_res
        # ls = lambda x, y: dxdy.cpu() - points_res.cpu()
        # outs['loss'] = dxdy.cpu() - points_res.cpu()
        outs["loss"] = dxdy.to(device) - points_res.to(device)
        outs["patches"] = patches
        return outs

    @staticmethod
    def flatten_64to1(semi, cell_size=8):
        """
        input:
            semi: tensor[batch, cell_size*cell_size, Hc, Wc]
            (Hc = H/8)
        outpus:
            heatmap: tensor[batch, 1, H, W]
        """
        from src.ultrapoint.utils.d2s import DepthToSpace

        depth2space = DepthToSpace(cell_size)
        heatmap = depth2space(semi)
        return heatmap

    @staticmethod
    def heatmap_nms(heatmap, nms_dist=4, conf_thresh=0.015):
        """
        input:
            heatmap: np [(1), H, W]
        """
        from src.ultrapoint.utils.utils import getPtsFromHeatmap

        heatmap = heatmap.squeeze()
        pts_nms = getPtsFromHeatmap(heatmap, conf_thresh, nms_dist)
        semi_thd_nms_sample = numpy.zeros_like(heatmap)
        semi_thd_nms_sample[pts_nms[1, :].astype(int), pts_nms[0, :].astype(int)] = 1
        return semi_thd_nms_sample

    @staticmethod
    def input_to_imgDict(sample, tb_images_dict):
        for e in list(sample):
            element = sample[e]
            if type(element) is torch.Tensor:
                if element.dim() == 4:
                    tb_images_dict[e] = element
                # print("shape of ", i, " ", element.shape)
        return tb_images_dict
