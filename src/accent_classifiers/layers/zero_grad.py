"""
Returns the input unchanged in the forward pass, and in the backward pass returns a
tensor of the same size as the gradient, but filled with zeros. This means that the
gradient will not be backpropagated.
"""


import torch


class GradZero(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return torch.zeros_like(grad_output)


def grad_zero(x):
    return GradZero.apply(x)
