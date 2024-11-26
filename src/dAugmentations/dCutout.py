import numpy as np
import torch
from torch import nn

from src.dAugmentations.dAugmentation import DAugmentation


# https://github.com/uoguelph-mlrg/Cutout/blob/master/util/cutout.py
class Cutout(nn.Module):
    """Randomly mask out one or more patches from an image.
    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """

    def __init__(self, n_holes, length, mean, noise, device=None):
        super(Cutout, self).__init__()
        self.n_holes = n_holes
        self.length = length
        self.mean = mean
        self.noise = noise
        self.device = device

    def forward(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        h = img.size(-2)
        w = img.size(-1)
        is_batch = len(img.shape) == 4

        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1:y2, x1:x2] = 0.0

        mask = torch.from_numpy(mask).to(self.device)
        mask = mask.expand_as(img)
        img = img * mask

        if self.noise:
            noise = (torch.randn_like(mask) * (1 - mask)).to(self.device)
            img = img + noise
        else:
            if is_batch:
                mean_v = self.mean.view(1, -1, 1, 1).to(self.device)
            else:
                mean_v = self.mean.view(-1, 1, 1).to(self.device)
            mean_like = (torch.ones_like(mask) * (1 - mask) * mean_v).to(self.device)
            img = img + mean_like

        return img


class DCutout(nn.Module, DAugmentation):
    def __init__(
        self,
        n_holes,
        length,
        mean,
        noise,
        low_m=None,
        high_m=None,
        det_m=None,
        learnable_app_prob=None,
        device=None,
    ):
        super(DCutout, self).__init__()
        self.daug_param_init(
            "Cutout",
            low_m=0.5,
            high_m=0.5,
            det_m=None,
            learnable_app_prob=learnable_app_prob,
            device=device,
        )
        self.n_holes = n_holes
        self.length = length
        self.mean = mean
        self.noise = noise

    def forward(self, input, augmented_idxs=None):
        b_size = input.shape[0] if self.stoch_batch_skips else 1

        # rsample application prob
        b_hard, b_soft = self.rsample_b((b_size,))

        # create augmentation with the sampled magnitude
        transform = Cutout(
            n_holes=self.n_holes,
            length=self.length,
            mean=self.mean,
            noise=self.noise,
            device=self.device,
        )

        # augmentation application
        out = self.aug_transform(
            input, transform, b_hard, b_soft, augmented_idxs, straight_through=True
        )
        return out
