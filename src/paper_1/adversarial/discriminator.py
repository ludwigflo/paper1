from torch.autograd import Function
from torch.autograd import Variable
from torch.nn import Module
from torch import tensor
import torch.nn as nn
import torch


# noinspection PyMethodOverriding
class RevGrad(Function):
    @staticmethod
    def forward(ctx, input_, alpha_):
        alpha = Variable(torch.Tensor([alpha_.item()]))
        device = input_.device
        alpha = alpha.to(device)
        ctx.save_for_backward(input_, alpha)
        output = input_
        return output

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = None
        _, alpha_ = ctx.saved_tensors
        if ctx.needs_input_grad[0]:
            grad_input = -grad_output * alpha_
        return grad_input, None


class GradientReversalLayer(Module):
    def __init__(self):
        """
        A gradient reversal layer.
        This layer has no parameters, and simply reverses the gradient
        in the backward pass.
        """
        super().__init__()
        self.gradient_reversal = RevGrad()

    def forward(self, data, gamma):
        return self.gradient_reversal.apply(data, gamma)


class Discriminator(Module):
    def __init__(self, feature_dim: int):
        super().__init__()

        self.gradient_reversal = GradientReversalLayer()

        self.params = nn.Sequential(
            nn.Linear(feature_dim, int(feature_dim/2), bias=True),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Linear(int(feature_dim/2), 1)
        )

    def forward(self, features: torch.Tensor, gamma: float) -> torch.Tensor:
        """"""

        # apply the gradient reversal layer
        features = self.gradient_reversal(features, gamma)

        # compute the discriminator logits
        logits = self.params(features)
        return logits
