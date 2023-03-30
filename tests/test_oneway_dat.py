"""
Test whether the OneWay DAT mode works. In this mode, only the gradients of accented
speech are reversed. This means that the classifier encourages the ASR encoder to keep
representing standard speech as standard speech, but also to map accented speech to
standard speech. This means:
- for a standard speech input, the loss should decrease when iterated over,
- for an accented speech input, the loss should increase when iterated over.
"""

import torch
from base import BaseTestClass


class TestOneWayDAT(BaseTestClass):
    def test_dat_optim_asr(self):
        self.config.ensemble.action = "train"
        self.config.ensemble.mode = "OneWayDAT"
        self._init_model()
        for optim_asr in [True, False]:
            with self.subTest(msg=f"optim only ASR: {optim_asr}"):
                return self._loss_change(optim_asr)

    def _loss_change(self, optim_asr):
        if optim_asr is True:
            self.optim = torch.optim.Adam(self.model.asr.parameters(), 0.001)
        for dl_idx, dataloader in enumerate(self.dataloaders):
            self.model.dataloader_idx = dl_idx
            func = self.assertLessEqual if dl_idx == 0 else self.assertGreaterEqual
            for i, batch in enumerate(dataloader):
                signal, signal_len, _, _ = batch
                losses = list()
                for _ in range(3):
                    self.model.asr.forward(
                        input_signal=signal, input_signal_length=signal_len
                    )
                    loss = self.model._ac_loss(dl_idx)
                    self.optim.zero_grad()
                    loss.backward()
                    self.optim.step()
                    if len(losses) > 0:
                        func(loss.item(), losses[-1], msg=losses)
                    losses.append(loss.item())
                if i == 2:
                    break  # break after three batches of each dl
            if dl_idx == 1:
                break  # break after 2 dls (one standard, one accented)
