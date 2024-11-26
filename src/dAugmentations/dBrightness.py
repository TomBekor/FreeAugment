import kornia as K
import torch
from torch import nn

from src.dAugmentations.dAugmentation import DAugmentation


class DBrightness(nn.Module, DAugmentation):
    def __init__(
        self,
        low_m=None,
        high_m=None,
        det_m=None,
        learnable_app_prob=None,
        stoch_batch_mag=True,
        device=None,
    ):
        super(DBrightness, self).__init__()
        self.daug_param_init(
            "Brightness", low_m, high_m, det_m, learnable_app_prob, device
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

        # create augmentation with the sampled magnitude
        clamped_param = torch.clamp(m, min=-1.0, max=1.0)
        transform = K.enhance.AdjustBrightness(brightness_factor=clamped_param)

        # augmentation application
        out = self.aug_transform(input, transform, b_hard, b_soft, augmented_idxs)
        return out
