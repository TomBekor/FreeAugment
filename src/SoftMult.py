import torch


class SoftMultFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, soft, hard, input):
        ctx.save_for_backward(soft, input)
        out = hard * input  # soft * hard
        return out

    @staticmethod
    def backward(ctx, grad_output):
        soft, input = ctx.saved_tensors
        return input * grad_output, input * grad_output, soft * grad_output
