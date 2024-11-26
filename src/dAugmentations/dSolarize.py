import torch
from torch import nn

from src.dAugmentations.dAugmentation import DAugmentation


class SolarizeFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, treshold):
        ctx.save_for_backward(input, treshold)
        input_shape = input.shape
        out = input.view(input_shape[0], -1)
        out = torch.where(out < treshold, out, 1.0 - out)
        out = out.view(*input_shape)
        out = torch.clamp(out, min=0.0, max=1.0)
        return out

    @staticmethod
    def backward(ctx, grad_output):

        # input image grad
        input, treshold = ctx.saved_tensors
        input_shape = input.shape
        flat_input = input.view(input_shape[0], -1)
        flat_grad = torch.where(flat_input < treshold, 1.0, -1.0)
        input_grad = grad_output * (flat_grad.view(*input_shape))

        # treshold grad
        flat_grad_output = grad_output.view(input_shape[0], -1)
        tresh_grad = flat_grad_output.sum(dim=-1, keepdim=True)

        return input_grad, tresh_grad


class Solarize(nn.Module):
    def __init__(self, tresholds):
        super(Solarize, self).__init__()
        self.tresholds = tresholds

    def forward(self, input):
        return SolarizeFunction.apply(input, self.tresholds)


class DSolarize(nn.Module, DAugmentation):
    def __init__(
        self,
        low_m=None,
        high_m=None,
        det_m=None,
        learnable_app_prob=None,
        stoch_batch_mag=True,
        device=None,
    ):
        super(DSolarize, self).__init__()
        self.daug_param_init(
            "Solarize", low_m, high_m, det_m, learnable_app_prob, device
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
        if self.stoch_batch_mag:  # patch
            m = m.unsqueeze(1)
        transform = Solarize(tresholds=m)

        # augmentation application
        out = self.aug_transform(input, transform, b_hard, b_soft, augmented_idxs)
        return out
