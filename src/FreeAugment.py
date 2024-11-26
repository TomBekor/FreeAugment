import torch
from torch import nn
from torch.nn.functional import gumbel_softmax
from torch.nn.parameter import Parameter

import wandb
from .AugLayer import *
from .sinkhorn_ops import *
from .SoftMult import SoftMultFunction
from .utils import *


class FreeAugment(nn.Module):
    def __init__(self, k_d_augmentations, device):
        super(FreeAugment, self).__init__()
        self.k = len(k_d_augmentations)
        self.k_d_augmentations = k_d_augmentations
        self.k_AugLayers = nn.ModuleList(
            [AugLayer(d_augmentation, device) for d_augmentation in k_d_augmentations]
        )
        self.device = device

        self.betas = Parameter(torch.log(torch.ones(self.k + 1) * 1 / (self.k + 1)))

        self.depth_temp = torch.tensor(1.0)
        self.aug_cat_g_sinkhorn_temp = torch.tensor(1.0)
        self.aug_cat_g_softmax_temp = torch.tensor(1.0)

        self.sinkhorn_iters = wandb.config["augs_gumbel_sinkhorn"]["iters"]
        self.log_epsilon = wandb.config["augs_gumbel_sinkhorn"]["log_epsilon"]

        self.apply_sinkhorn = wandb.config["apply_sinkhorn"]
        self.apply_depth = wandb.config["apply_depth"]

        self.stoch_batch_aug = wandb.config["stoch_batch_aug"]
        self.stoch_batch_depth = wandb.config["stoch_batch_depth"]

        self.apply_chosen_augs_only = wandb.config["apply_chosen_augs_only"]
        self.calc_soft_mult_grads = wandb.config["calc_soft_mult_grads"]

    def forward(self, input):

        batch_size = input.shape[0]

        # for each layer, sample soft categorical samples from relaxed categorical distributions
        categorical_logits_matrix = self.categoricals_matrix(batch_size)

        # selects augmentations given the logits calculated above.
        # can use either: softmax / sinkhorn. (i.e. regular sampling vs. learning depth)
        hard_and_soft_categorical_samples_tup = self.choose_augs(
            logits_mat=categorical_logits_matrix, hard=True
        )

        # save layer outs for depth learning
        layer_outs = [input]

        # iterate over pairs of augmentation layers and chosen augmentation for each layer
        (
            hard_categorical_samples,
            soft_categorical_samples,
        ) = hard_and_soft_categorical_samples_tup
        hard_categorical_samples = hard_categorical_samples.permute(
            1, 0, 2
        )
        soft_categorical_samples = soft_categorical_samples.permute(
            1, 0, 2
        )

        augmented_idxs = None

        if self.apply_depth:
            # choose augmentation depth
            # if apply_depth=False, the last entry of the
            # resulted oh vector will be on: [0,...,0,1]
            betas_hat, betas_soft = self.sample_betas(batch_size)

            if self.apply_chosen_augs_only:
                with torch.no_grad():
                    augmented_idxs = torch.einsum(
                        "ijk, ij -> ijk", hard_categorical_samples, betas_hat[1:]
                    )
        else:
            if self.apply_chosen_augs_only:
                augmented_idxs = hard_categorical_samples

        for i, (layer, oh_augmentation, soft_augs) in enumerate(
            zip(self.k_AugLayers, hard_categorical_samples, soft_categorical_samples)
        ):

            # set idxs to None if we want to apply all of the augmentations
            layer_augmented_idxs = (
                None if not self.apply_chosen_augs_only else augmented_idxs[i]
            )

            # apply layer with chosen augmentations
            layer_out = layer(
                layer_outs[-1],
                aug_oh=oh_augmentation,
                soft_augs=soft_augs,
                layer_augmented_idxs=layer_augmented_idxs,
            )

            # save layer out for depth learning
            layer_outs.append(layer_out)

        if self.apply_depth:
            # apply depth selection for each image on the batch
            out = torch.zeros_like(input)
            for i, (oh, soft) in enumerate(zip(betas_hat, betas_soft)):
                if self.calc_soft_mult_grads:
                    out += SoftMultFunction.apply(
                        soft.view(-1, 1, 1, 1), oh.view(-1, 1, 1, 1), layer_outs[i]
                    )
                else:
                    out += oh.view(-1, 1, 1, 1) * layer_outs[i]
        else:
            out = layer_outs[-1]

        return out

    def categoricals_matrix(self, batch_size):

        cat_logits_list = [layer.augs_categorical_dist for layer in self.k_AugLayers]
        cat_logits = torch.vstack(cat_logits_list)

        if self.stoch_batch_aug:
            cat_logits = cat_logits.repeat(batch_size, 1, 1)
        else:
            cat_logits = cat_logits.unsqueeze(0)

        return cat_logits

    def choose_augs(self, logits_mat, hard=True, dim=-1):

        if self.apply_sinkhorn:

            # gumbel-sinkhorn

            sink, log_alpha_w_noise = my_new_gumbel_sinkhorn(
                log_alpha=logits_mat,
                temp=self.aug_cat_g_sinkhorn_temp,
                n_iters=self.sinkhorn_iters,
                log_epsilon=self.log_epsilon,
                device=self.device,
            )

            if hard:

                index = sink.max(dim, keepdim=True)[1]
                hard_samples_no_grad = torch.zeros_like(
                    sink, memory_format=torch.legacy_contiguous_format
                ).scatter_(dim, index, 1.0)
                hard_minus_sink = hard_samples_no_grad - sink
                hard_samples_w_grad = hard_minus_sink.detach() + sink

                ret = hard_samples_w_grad
                soft = sink

                assert (
                    torch.sum(ret).item() == ret.shape[0] * ret.shape[1]
                ), "Sinkhorn didn't choose k augmentations for each image"

            else:
                ret = sink
                soft = sink

        else:
            # gumbel-softmax
            y_soft = gumbel_softmax(
                logits_mat, tau=self.aug_cat_g_softmax_temp, hard=hard, dim=dim
            )
            # Straight through
            index = y_soft.max(dim, keepdim=True)[1]
            y_hard = torch.zeros_like(
                logits_mat, memory_format=torch.legacy_contiguous_format
            ).scatter_(dim, index, 1.0)
            ret = (y_hard - y_soft).detach() + y_soft
            soft = y_soft

        return ret, soft
    
    def sample_betas(self, batch_size):
        betas = self.betas
        if self.stoch_batch_depth:
            betas = betas.repeat(batch_size, 1)
        else:
            betas = betas.unsqueeze(0)

        dim = -1
        betas_soft = gumbel_softmax(
            logits=betas, tau=self.depth_temp, hard=False, dim=dim
        )
        index = betas_soft.max(dim, keepdim=True)[1]
        betas_hard = torch.zeros_like(
            betas, memory_format=torch.legacy_contiguous_format
        ).scatter_(dim, index, 1.0)
        betas_hard_minus_soft = betas_hard - betas_soft
        betas_hat = betas_hard_minus_soft.detach() + betas_soft

        betas_hat = betas_hat.transpose(0, 1)
        betas_soft = betas_soft.transpose(0, 1)

        if self.stoch_batch_depth:
            assert (
                torch.sum(betas_hat).item() == batch_size
            ), "Depth choosing skipped some images along batch"
        else:
            assert (
                torch.sum(betas_hat).item() == 1
            ), "Depth choosing skipped some images along batch"

        return betas_hat, betas_soft

    def set_depth_temp(self, new_temp):
        self.depth_temp = torch.tensor(new_temp)

    def set_aug_cat_g_sinkhorn_temp(self, new_temp):
        self.aug_cat_g_sinkhorn_temp = torch.tensor(new_temp)

    def set_aug_cat_g_softmax_temp(self, new_temp):
        self.aug_cat_g_softmax_temp = torch.tensor(new_temp)

    def set_apply_chosen_augs_only(self, new_val):
        self.apply_chosen_augs_only = new_val
        for layer in self.k_AugLayers:
            layer.apply_chosen_augs_only = new_val

    def set_calc_soft_mult_grads(self, new_val):
        self.calc_soft_mult_grads = new_val
        for layer in self.k_AugLayers:
            layer.set_calc_soft_mult_grads(new_val)

    def get_params(self):
        return [auglayer.get_params() for auglayer in self.k_AugLayers]

    def get_cat_dist(self):
        return [auglayer.get_cat_dist() for auglayer in self.k_AugLayers]

    def get_augs_names(self):
        # augs names at the first layer
        return self.k_AugLayers[0].get_aug_names()

    def __repr__(self) -> str:
        return self.k_AugLayers.__repr__()
