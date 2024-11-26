from torch import nn

from src.dAugmentations.dAugmentation import DAugmentation


class AutoContrast(nn.Module):
    def __init__(self):
        super(AutoContrast, self).__init__()

    def forward(self, input):
        min_val = input.amin(dim=(-2, -1), keepdim=True)
        max_val = input.amax(dim=(-2, -1), keepdim=True)
        return (input - min_val) / (max_val - min_val + 1e-6)


class DAutoContrast(nn.Module, DAugmentation):
    def __init__(
        self, low_m=None, high_m=None, det_m=None, learnable_app_prob=None, device=None
    ):
        super(DAutoContrast, self).__init__()
        self.daug_param_init(
            "AutoContrast",
            low_m=0.5,
            high_m=0.5,
            det_m=None,
            learnable_app_prob=learnable_app_prob,
            device=device,
        )

    def forward(self, input, augmented_idxs=None):
        b_size = input.shape[0] if self.stoch_batch_skips else 1

        # rsample magnitude and application prob
        b_hard, b_soft = self.rsample_b((b_size,))

        # create augmentation with the sampled magnitude
        transform = AutoContrast()

        # augmentation application
        out = self.aug_transform(input, transform, b_hard, b_soft, augmented_idxs)
        return out
