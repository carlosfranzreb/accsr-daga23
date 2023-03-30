"""From https://discuss.pytorch.org/t/solved-reverse-gradients-in-backward-pass/3589/3"""


from torch.autograd import Function


class GradReverse(Function):
    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg()


def grad_reverse(x):
    return GradReverse.apply(x)
