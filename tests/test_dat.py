"""Test whether the gradient reversal works."""

import torch
from base import BaseTestClass


class TestDAT(BaseTestClass):
    def test_dat_optim_asr(self):
        self.config.ensemble.action = "train"
        self.config.ensemble.mode = "DAT"
        self._init_model()
        for optim_asr in [True, False]:
            with self.subTest(msg=f"optim only ASR: {optim_asr}"):
                return self._loss_change(self.assertGreaterEqual, optim_asr)

    def test_mtl_optim_asr(self):
        self.config.ensemble.action = "train"
        self.config.ensemble.mode = "MTL"
        self._init_model()
        for optim_asr in [True, False]:
            with self.subTest(msg=f"optim only ASR: {optim_asr}"):
                return self._loss_change(self.assertLessEqual, optim_asr)

    def _loss_change(self, func, optim_asr):
        if optim_asr is True:
            self.optim = torch.optim.Adam(self.model.asr.parameters(), 0.01)
        for i, batch in enumerate(self.dataloaders[0]):
            signal, signal_len, _, _ = batch
            losses = list()
            for _ in range(3):
                self.model.asr.forward(
                    input_signal=signal, input_signal_length=signal_len
                )
                loss = self.model._ac_loss(0)
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()
                if len(losses) > 0:
                    func(loss.item(), losses[-1], msg=losses)
                losses.append(loss.item())
            if i == 3:
                break
