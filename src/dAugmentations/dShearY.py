import kornia as K
import torch
from torch import nn

from src.dAugmentations.dAugmentation import DAugmentation


class DShearY(nn.Module, DAugmentation):
    def __init__(
        self,
        low_m=None,
        high_m=None,
        det_m=None,
        learnable_app_prob=None,
        stoch_batch_mag=True,
        device=None,
    ):
        super(DShearY, self).__init__()
        self.daug_param_init("ShearY", low_m, high_m, det_m, learnable_app_prob, device)
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
        zero_shear = torch.tensor(0.0).repeat(m_size).to(self.device)
        shear = torch.vstack([zero_shear, m]).T
        transform = K.geometry.transform.Shear(shear=shear)

        # augmentation application
        out = self.aug_transform(input, transform, b_hard, b_soft, augmented_idxs)
        return out
