import kornia as K
import torch
from torch import nn

from src.dAugmentations.dAugmentation import DAugmentation


class Sharpness(nn.Module):
    def __init__(self, sharpness_factor):
        super(Sharpness, self).__init__()
        self.sharpness_factor = sharpness_factor

    def forward(self, input):
        return K.enhance.sharpness(input, factor=self.sharpness_factor)


class DSharpness(
    nn.Module, DAugmentation
):
    def __init__(
        self,
        low_m=None,
        high_m=None,
        det_m=None,
        learnable_app_prob=None,
        stoch_batch_mag=True,
        device=None,
    ):
        super(DSharpness, self).__init__()
        self.daug_param_init(
            "Sharpness", low_m, high_m, det_m, learnable_app_prob, device
        )
        self.stoch_batch_mag = stoch_batch_mag

    def forward(self, input, augmented_idxs=None):
        m_size = input.shape[0] if self.stoch_batch_mag else 1
        b_size = input.shape[0] if self.stoch_batch_skips else 1

        if augmented_idxs is not None and self.stoch_batch_mag:
            m_size = int(augmented_idxs.sum())

        # rsample magnitude and application prob
        m = self.rsample_m((m_size,))
        b_hard, b_soft = self.rsample_b((b_size,))

        if 1.0 in m:
            print("Sharpness with m=1.!!!")

        # create augmentation with the sampled magnitude
        clamped_sharpness_factor = torch.clamp(m, min=0.0)
        transform = Sharpness(sharpness_factor=clamped_sharpness_factor)

        # augmentation application
        out = self.aug_transform(input, transform, b_hard, b_soft, augmented_idxs)
        return out
