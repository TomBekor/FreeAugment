import kornia as K
from torch import nn

from src.dAugmentations.dAugmentation import DAugmentation


class DEqualize(nn.Module, DAugmentation):
    def __init__(
        self, low_m=None, high_m=None, det_m=None, learnable_app_prob=None, device=None
    ):
        super(DEqualize, self).__init__()
        self.daug_param_init(
            "Equalize",
            low_m=0.5,
            high_m=0.5,
            det_m=None,
            learnable_app_prob=learnable_app_prob,
            device=device,
        )

    def forward(self, input, augmented_idxs=None):
        b_size = input.shape[0] if self.stoch_batch_skips else 1

        # rsample application prob
        b_hard, b_soft = self.rsample_b((b_size,))

        # create augmentation with the sampled magnitude
        transform = K.enhance.equalize

        # augmentation application
        out = self.aug_transform(
            input, transform, b_hard, b_soft, augmented_idxs, straight_through=True
        )
        return out
