from torch import nn


class LinearLayer(nn.Module):
    """Dense layer with a optional activation function and dropout."""

    def __init__(self, n_input, n_output, activation_fn, dropout):
        """Activation function and dropout can be null."""
        super().__init__()
        self.dropout = nn.Dropout(dropout) if dropout is not None else None
        self.layer = nn.Linear(n_input, n_output)
        self.activation_fn = activation_fn

    def forward(self, x):
        x = self.dropout(x) if self.dropout is not None else x
        out = self.layer(x)
        return self.activation_fn(out) if self.activation_fn is not None else out
