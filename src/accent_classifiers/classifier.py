from torch import nn
import torch.nn.functional as F

from src.accent_classifiers.layers.linear import LinearLayer
from src.accent_classifiers.layers.reverse_grad import grad_reverse


class AC(nn.Module):
    """Fully-connected neural network to classify accents."""

    def __init__(self, n_accents, dropout, mode, standard):
        """Dropout can be null. `standard` is the standard accent for OneWayDAT. Other
        accents are reversed; `standard` is not. DAT mode ignores this `standard` arg
        and reverse all gradients."""
        super().__init__()
        self.mode = mode
        self.standard = standard
        self.fc1 = LinearLayer(512, 1024, F.relu, dropout)
        self.fc2 = LinearLayer(1024, 256, F.relu, dropout)
        self.fc3 = LinearLayer(256, n_accents, None, dropout)

    def forward(self, x, y):
        if self.mode == "DAT":
            x = grad_reverse(x)
        elif self.mode == "OneWayDAT":
            if y != self.standard:
                x = grad_reverse(x)
        x = x.mean(dim=1)
        x = self.fc1(x)
        x = self.fc2(x)
        return self.fc3(x)
