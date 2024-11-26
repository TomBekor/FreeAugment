import kornia as K
import torch
from torch import nn

from src.dAugmentations.dAugmentation import DAugmentation


class DPosterize(nn.Module, DAugmentation):
    def __init__(
        self,
        low_m=None,
        high_m=None,
        det_m=None,
        learnable_app_prob=None,
        stoch_batch_mag=True,
        device=None,
    ):
        super(DPosterize, self).__init__()
        self.daug_param_init(
            "Posterize", low_m, high_m, det_m, learnable_app_prob, device
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
        round_m = (torch.round(m) - m).detach() + m
        transform = lambda x: K.enhance.posterize(x, bits=round_m)

        # augmentation application
        out = self.aug_transform(
            input, transform, b_hard, b_soft, augmented_idxs, straight_through=True
        )

        # magnitude straight-trough
        if augmented_idxs is None:
            st_m = round_m
        else:
            if self.stoch_batch_mag:
                st_m = torch.zeros(out.shape[0]).to(self.device)
                st_m[augmented_idxs == 1.0] = round_m
            else:
                st_m = torch.zeros_like(round_m).to(self.device)
                if torch.all(augmented_idxs == 1.0):
                    st_m = round_m
        out = out + st_m.view(-1, 1, 1, 1) - st_m.view(-1, 1, 1, 1).detach()

        return out
