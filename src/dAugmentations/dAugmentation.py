import torch
from torch.distributions.relaxed_bernoulli import RelaxedBernoulli
from torch.distributions.uniform import Uniform
from torch.nn.parameter import Parameter

import wandb
from src.aug_config import *
from src.SoftMult import SoftMultFunction


class DAugmentation:

    image_size = (0, 0)

    app_prob = torch.tensor(1.0)  # 1 - skip_prob = application_prob

    def __init__(self):
        pass

    def daug_param_init(
        self, aug_name, low_m, high_m, det_m, learnable_app_prob, device
    ):

        self.device = device
        self.aug_name = aug_name

        self.mag_sigmoid_temp = 0.1

        self.calc_soft_mult_grads = wandb.config["calc_soft_mult_grads"]

        self.learn_skips = wandb.config["learn_skips"]
        self.stoch_batch_skips = wandb.config["stoch_batch_skips"]
        self.b_temperature = Parameter(torch.tensor([2.2]), requires_grad=False)

        if low_m is not None and high_m is not None and det_m is not None:
            raise Exception(
                "DAugmentations can have (low_m,high_m) or (det_m) - not both."
            )
        elif low_m is not None and high_m is not None:
            self.low_m = Parameter(self.inverse_temp_sigmoid(torch.tensor(low_m)))
            self.high_m = Parameter(self.inverse_temp_sigmoid(torch.tensor(high_m)))
            self.stoch_m_flag = True
        elif det_m is not None:
            self.det_m = Parameter(self.inverse_temp_sigmoid(torch.tensor(det_m)))
            self.stoch_m_flag = False
        else:
            raise Exception("(low_m,high_m) or (det_m) are not defined.")

        self.app_prob = DAugmentation.app_prob.to(device)
        if self.learn_skips:
            self.app_prob = Parameter(self.app_prob)

        self.verbose = False
        self.stoch_batch_mag = False
        self.mult_factor = 1.0
        self.init_aug_bounds()

        self.stoch_mag_along_batch = wandb.config["stoch_batch_mag"]

        self.method = "uniform"
        # self.method = 'normal'

    def init_aug_bounds(self):
        if self.aug_name in aug_bounds:
            self.low_bound = aug_bounds[self.aug_name][0] * self.mult_factor
            self.high_bound = aug_bounds[self.aug_name][1] * self.mult_factor

    def rsample_m(self, size=1):
        ### rsample magnitude from learnable magnitude range
        ### rsample meaning - sample using a reparameterization trick

        if self.method == "uniform":

            with torch.no_grad():
                if not self.low_m < self.high_m:
                    temp = self.low_m.data
                    self.low_m.data = self.high_m.data
                    self.high_m.data = temp + 1e-4

            if self.stoch_m_flag:
                if self.low_m == self.high_m:
                    return self.low_m.repeat(size)
                elif not self.low_m < self.high_m:  # if not
                    uniform_dist = Uniform(self.high_m, self.low_m)
                else:
                    uniform_dist = Uniform(self.low_m, self.high_m)
                m = uniform_dist.rsample(size)

            else:
                m = self.temp_sigmoid(self.det_m)

            # scale to unnormalized augmentation magnitudes
            m = (
                self.temp_sigmoid(m) * (self.high_bound - self.low_bound)
            ) + self.low_bound

            assert (
                m.shape == size
            ), f"sampled_m.size={m.shape}, while requested size is {size}. \ndetails: {self.low_m, self.high_m, self.det_m} \n{self}"

            if not self.stoch_mag_along_batch:
                m = m[0]

            if self.verbose:
                print(f"magnitude: {m}")
            return m

        elif self.method == "normal":

            raise Exception

            # ---------Normal-------- #
            # -
            # Low_m stands for Std
            # High_m stands for Mean
            # -
            # ----------------------- #

            if self.stoch_m_flag:
                z = torch.randn(size).to(self.device)
                m = (self.temp_sigmoid(self.low_m) * z) + self.temp_sigmoid(
                    self.high_m
                )  # reparameterization trick
            else:
                m = self.temp_sigmoid(self.det_m)

            # scale to unnormalized augmentation magnitudes
            m = (m * (self.high_bound - self.low_bound)) + self.low_bound

            assert (
                m.shape == size
            ), f"sampled_m.size={m.shape}, while requested size is {size}. \ndetails: {self.low_m, self.high_m, self.det_m} \n{self}"

            if not self.stoch_mag_along_batch:
                m = m[0]

            if self.verbose:
                print(f"magnitude: {m}")
            return m

    def rsample_b(self, size):
        ### rsample application probability from learnable logit
        ### rsample meaning - sample using a reparameterization trick
        ### the main trick for `hard sampling` is to do `y_hard - y_soft.detach() + y_soft` as
        ### in hard rsampling from gumbel_softmax.
        if not self.learn_skips:
            self.app_prob = DAugmentation.app_prob.to(self.device)
        clamped_prob = torch.clamp(self.app_prob, min=0.0, max=1.0)

        if clamped_prob == 1.0 or clamped_prob == 0.0:
            b_hard = clamped_prob.repeat(size)
            b_soft = b_hard
        else:
            bernoulli_dist = RelaxedBernoulli(
                temperature=self.b_temperature, probs=clamped_prob
            )
            b_soft = bernoulli_dist.rsample(size)
            b_hard = torch.round(b_soft)
            b_hard_minus_soft = b_hard - b_soft
            b_hard = b_hard_minus_soft.detach() + b_soft

        return b_hard, b_soft

    def aug_transform(
        self,
        input,
        transform,
        b_hard,
        b_soft,
        augmented_idxs=None,
        straight_through=False,
    ):

        if augmented_idxs is not None:
            selected_idxs = torch.logical_and(augmented_idxs == 1.0, b_hard == 1.0)
        else:
            selected_idxs = torch.ones(input.shape[0], dtype=torch.bool)  # all True
        neg_selected_idxs = selected_idxs == False

        if torch.all(neg_selected_idxs):
            return input

        transformed_input = transform(input[selected_idxs])
        if straight_through:
            transformed_input = (
                transformed_input - input[selected_idxs]
            ).detach() + input[selected_idxs]

        out = torch.zeros_like(input)

        if torch.all(selected_idxs):
            if self.calc_soft_mult_grads:
                out += SoftMultFunction.apply(
                    b_soft.view(-1, 1, 1, 1),
                    b_hard.view(-1, 1, 1, 1),
                    transformed_input,
                )
                out += SoftMultFunction.apply(
                    1 - b_soft.view(-1, 1, 1, 1), 1 - b_hard.view(-1, 1, 1, 1), input
                )
            else:
                out += b_hard.view(-1, 1, 1, 1) * transformed_input
                out += (1 - b_hard.view(-1, 1, 1, 1)) * input
            return out
        else:
            assert (
                not self.calc_soft_mult_grads
            ), "Calc gumbel grads and not applying all of the augmentations."
            out[selected_idxs] += (
                b_hard[selected_idxs].view(-1, 1, 1, 1) * transformed_input
            )
            out[neg_selected_idxs] += input[
                neg_selected_idxs
            ]  # grads won't populate to b_hard[neg_selected_idxs]
            return out

    def temp_sigmoid(self, vals):
        return torch.sigmoid(self.mag_sigmoid_temp * vals)

    def inverse_temp_sigmoid(self, y):
        inverse_sigmoid = lambda y: torch.log(y / (1 - y))
        return inverse_sigmoid(y) / self.mag_sigmoid_temp

    @staticmethod
    def set_skip_prob(new_skip_prob):
        assert not wandb.config[
            "learn_skips"
        ], "Can't change/schedule app_prob value while learning it"
        DAugmentation.app_prob = torch.tensor(
            1.0 - new_skip_prob
        )  # 1 - skip_prob = application_prob

    def get_params(self):
        with torch.no_grad():
            params = []

            if self.stoch_m_flag:
                params.append(self.temp_sigmoid(self.low_m).detach().item())
                params.append(self.temp_sigmoid(self.high_m).detach().item())
            else:
                params.append(self.temp_sigmoid(self.det_m).detach().item())

            params.append(self.app_prob.detach().item())

            return params

    def __str__(self) -> str:
        s = f"{self.aug_name}:"
        if self.stoch_m_flag:
            s += f" magnitude:[{self.low_m},{self.high_m}]"
        else:
            s += f" magnitude:({self.det_m})"
        s += f" app_prob:({self.app_prob})"
        return s
