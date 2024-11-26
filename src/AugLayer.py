import torch
from torch import nn
from torch.nn.parameter import Parameter

import wandb
from .SoftMult import SoftMultFunction
from .utils import *


class AugLayer(nn.Module):
    def __init__(self, d_augmentations, device):
        super(AugLayer, self).__init__()
        self.d_augmentations = nn.ModuleList(d_augmentations)
        with torch.no_grad():
            uniform_cat_dist = torch.log(
                torch.ones(len(self.d_augmentations)) / len(self.d_augmentations)
            )
        self.augs_categorical_dist = Parameter(uniform_cat_dist)
        self.device = device
        self.stoch_batch_aug = wandb.config["stoch_batch_aug"]
        self.calc_soft_mult_grads = wandb.config["calc_soft_mult_grads"]

    def forward(
        self,
        input,
        aug_oh=None,
        soft_augs=None,
        layer_augmented_idxs=None,
        gumbel_temp=None,
        batch_size=1,
        verbose=False,
    ):

        batch_size = input.shape[0]

        if aug_oh is not None:
            one_hot = aug_oh
        else:
            raise NotImplementedError

        if self.stoch_batch_aug and len(one_hot.shape) == 1:
            assert False, "Missing augmentation samples for each image on batch"

        if len(one_hot.shape) == 1:
            one_hot = one_hot.repeat(batch_size, 1)
            soft_augs = soft_augs.repeat(batch_size, 1)

        out = torch.zeros_like(input)
        one_hot = one_hot.transpose(0, 1)
        soft_augs = soft_augs.transpose(0, 1)
        layer_augmented_idxs = (
            None
            if layer_augmented_idxs is None
            else layer_augmented_idxs.transpose(0, 1)
        )
        for i, (oh, soft, daug) in enumerate(
            zip(one_hot, soft_augs, self.d_augmentations)
        ):
            aug_augmented_idxs = (
                None if layer_augmented_idxs is None else layer_augmented_idxs[i]
            )

            if self.calc_soft_mult_grads:
                out += SoftMultFunction.apply(
                    soft.view(-1, 1, 1, 1),
                    oh.view(-1, 1, 1, 1),
                    daug(input, aug_augmented_idxs),
                )
            else:
                out += oh.view(-1, 1, 1, 1) * daug(input, aug_augmented_idxs)

        return out

    def get_params(self):
        return [daug.get_params() for daug in self.d_augmentations]

    def get_aug_names(self):
        return [daug.aug_name for daug in self.d_augmentations]

    def get_cat_dist(self):
        return self.augs_categorical_dist.detach().cpu().numpy()

    def get_daug(self, name):
        for daug in self.d_augmentations:
            if daug.aug_name == name:
                return daug

    def get_daug_params(self, name):
        for daug in self.d_augmentations:
            if daug.aug_name == name:
                return daug.parameters()
        return []

    def contains(self, name):
        for daug in self.d_augmentations:
            if daug.aug_name == name:
                return True
        return False

    def set_calc_soft_mult_grads(self, new_val):
        self.calc_soft_mult_grads = new_val
        for daug in self.d_augmentations:
            daug.calc_soft_mult_grads = new_val

    def __str__(self) -> str:
        s = f"{str(self.augs_categorical_dist)}\n"
        s += "Augmentations:\n"
        for daug in self.d_augmentations:
            s += f"{str(daug)}\n"
        return s

    def __repr__(self) -> str:
        with torch.no_grad():
            dist = nn.Softmax(dim=0)(self.augs_categorical_dist).tolist()
            dist = [round(p, 2) for p in dist]
            s = f"Categorical distribution: {dist}\n"
            s += "Augmentations:\n"
            for daug in self.d_augmentations:
                s += f"{str(daug)}\n"
        return s
