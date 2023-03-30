"""
Class that manages the accent classifiers (ACs). There is one AC per accent, and
for each input this class receives, it passes it to the appropriate AC. The other ACs
also receive the input and perform a training step on it, but it is detached
from the gradient graph that links this model to the ASR model. Therefore, the other
ACs are trained here. Each model has its optimizer, which is an Adam optimizer whose
learning rate is taken from the optimizer configuration.
"""


import torch
from torch.nn import CrossEntropyLoss
from torch.optim import Adam

from src.init_classifier import init_classifier


class SwitchDAT(torch.nn.Module):
    def __init__(self, config, seen_langs):
        """
        Initialize the ACs and the optimizers.
        - The ACs all have 0 as their standard accent. All other accents are mapped
        to 1. Therefore, the true label of the AC of the same accent is always zero,
        and the true label of all other ACs is always one.
        - The LR of the AC optimizers is taken from the optim config, which must be
        linked within the AC config.
        """
        if config.mode not in ("DAT", "MTL"):
            raise ValueError("Invalid mode. Must be DAT or MTL.")

        super().__init__()
        self.config = config

        # find the LR for the ACs; use the ensemble LR if there is no AC-specific LR
        lr = config.optim.lr
        if "ac_args" in config.optim:
            if "lr" in config.optim.ac_args:
                lr = config.optim.ac_args.lr

        # initialize the modules and optimizers, one of each per seen lang
        self.seen_langs = seen_langs
        self._modules, self.optimizers = dict(), dict()
        for i, lang in enumerate(seen_langs):
            self._modules[lang] = init_classifier(config, config.mode, standard=0)
            self.optimizers[lang] = Adam(self._modules[lang].parameters(), lr)

        # initialize the loss function
        self.loss_fn = CrossEntropyLoss()

    def forward(self, x, y):
        """
        1. Computes a training step on all the ACs whose accent is not `y`. Their
            true labels are always one, as the batch does not belong to their accents.
        2. Returns the output of the forward pass of the AC whose accent is `y`. This
            AC is part of the ensemble for this batch; its gradients will be
            backpropagated to the ASR model. True label is zero.
        - If the input `x` does not require a gradient, it comes from a validation
            step. For validation steps, skip step 1.
        """
        if y >= len(self.seen_langs):  # batch from unseen accent
            n_outs = 2 if self.config.binary else len(self.seen_langs)
            return torch.zeros((x.shape[0], n_outs))
        if x.requires_grad:  # the input comes from a training step
            y_other = torch.ones(x.shape[0], dtype=torch.int64, device=x.device)
            x_detached = x.detach()
            for i, lang in enumerate(self.seen_langs):
                if i != y:  # different accent; train the AC
                    self.optimizers[lang].zero_grad()
                    out = self._modules[lang].forward(x_detached, y)
                    loss = self.loss_fn(out, y_other)
                    loss.backward()
                    self.optimizers[lang].step()

        # return the output of the forward pass of the AC of the same accent
        return self._modules[self.seen_langs[y]].forward(x, 0)
