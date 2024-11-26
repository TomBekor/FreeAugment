import torch
from torch import nn

from src.dAugmentations.dAugmentation import DAugmentation


class Color(nn.Module):
    def __init__(self, coefs):
        super(Color, self).__init__()
        self.coefs = coefs.view(-1, 1, 1, 1)

    def forward(self, input):
        blended_image = self.coefs * input + (1 - self.coefs) * input.mean(
            1, keepdim=True
        )
        return torch.clamp(blended_image, min=0.0, max=1.0)


class DColor(nn.Module, DAugmentation):
    def __init__(
        self,
        low_m=None,
        high_m=None,
        det_m=None,
        learnable_app_prob=None,
        stoch_batch_mag=True,
        device=None,
    ):
        super(DColor, self).__init__()
        self.daug_param_init("Color", low_m, high_m, det_m, learnable_app_prob, device)
        self.stoch_batch_mag = stoch_batch_mag

    def forward(self, input, augmented_idxs=None):
        m_size = input.shape[0] if self.stoch_batch_mag else 1
        b_size = input.shape[0] if self.stoch_batch_skips else 1

        if augmented_idxs is not None and self.stoch_batch_mag:
            m_size = int(augmented_idxs.sum())

        # rsample magnitude and application prob
        m = self.rsample_m((m_size,))
        b_hard, b_soft = self.rsample_b((b_size,))

        # create augmentation with the sampled magnitude
        transform = Color(coefs=m)

        # augmentation application
        out = self.aug_transform(input, transform, b_hard, b_soft, augmented_idxs)
        return out
