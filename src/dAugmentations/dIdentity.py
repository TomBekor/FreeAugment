from torch import nn

from src.dAugmentations.dAugmentation import DAugmentation


class DIdentity(nn.Module, DAugmentation):
    def __init__(
        self, low_m=None, high_m=None, det_m=None, learnable_app_prob=None, device=None
    ):
        super(DIdentity, self).__init__()
        self.daug_param_init(
            "Identity",
            low_m=0.5,
            high_m=0.5,
            det_m=None,
            learnable_app_prob=learnable_app_prob,
            device=device,
        )

    def forward(self, input, augmented_idxs=None):
        return input
