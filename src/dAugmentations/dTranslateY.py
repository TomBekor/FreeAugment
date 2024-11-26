import kornia as K
import torch
from torch import nn

from src.dAugmentations.dAugmentation import DAugmentation


class DTranslateY(
    nn.Module, DAugmentation
):
    def __init__(
        self,
        low_m=None,
        high_m=None,
        det_m=None,
        learnable_app_prob=None,
        device=None,
        stoch_batch_mag=True,
        chunk_size=1000,
    ):
        super(DTranslateY, self).__init__()
        self.daug_param_init(
            "TranslateY", low_m, high_m, det_m, learnable_app_prob, device
        )

        self.stoch_batch_mag = stoch_batch_mag
        self.chunk_size = chunk_size

        # Set augmentation bounds that depends on image size - magnitude is in pixles
        self.mult_factor = DAugmentation.image_size[1]
        self.init_aug_bounds()

    def forward(self, input, augmented_idxs=None):
        m_size = input.shape[0] if self.stoch_batch_mag else 1
        b_size = input.shape[0] if self.stoch_batch_skips else 1

        if augmented_idxs is not None and self.stoch_batch_mag:
            m_size = int(augmented_idxs.sum())

        # rsample magnitude and application prob
        m = self.rsample_m((m_size,))
        b_hard, b_soft = self.rsample_b((b_size,))

        # create augmentation with the sampled magnitude
        zero_translate = torch.tensor(0.0).repeat(m_size).to(self.device)
        translation = torch.vstack([zero_translate, m]).T
        transform = K.geometry.transform.Translate(translation=translation)

        # augmentation application
        out = self.aug_transform(input, transform, b_hard, b_soft, augmented_idxs)
        return out
