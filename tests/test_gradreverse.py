"""Test whether the gradient reversal works in the accent classifier (AC) by freezing
the AC. Then, the negated gradient is backpropagated to the dummy ASR model, which
should result in the loss increasing. Test also the MTL version, where the gradient
is not negated; the loss should decrease in this case."""


import unittest
import torch
from src.accent_classifiers.classifier import AC


class TestGradReverse(unittest.TestCase):
    def setUp(self):
        self.asr = torch.nn.Sequential(
            torch.nn.Linear(512, 512),
            torch.nn.Linear(512, 512),
            torch.nn.Linear(512, 512),
        )
        self.optim = torch.optim.Adam(self.asr.parameters(), 0.001)
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.x = torch.rand((1, 512, 512))
        self.y = torch.tensor([0])

    def test_mtl(self):
        ac = AC(2, 0, "MTL", 0)
        self._run_steps(ac, self.assertLessEqual)

    def test_dat(self):
        ac = AC(2, 0, "DAT", 0)
        self._run_steps(ac, self.assertGreaterEqual)

    def _run_steps(self, ac, func, n=5):
        """Run n forward and backward passes, checking with `func` whether the loss
        changes in the desired direction. Only the parameters of the ASR model are
        optimized, which appear after the gradient reversal in the backprop."""
        losses = list()
        for _ in range(5):
            out_asr = self.asr.forward(self.x)
            out_ac = ac.forward(out_asr, self.y)
            loss = self.loss_fn(out_ac, self.y)
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()
            if len(losses) > 1:
                func(loss.item(), losses[-1])
            losses.append(loss.item())
