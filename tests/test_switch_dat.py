"""
Test whether SwitchDAT works. SwitchDAT is an adversary that comprises several ACs,
one per seen accent. Each AC has its accent as 'standard' and all others as 'accented'.
During training, each batch comprises a batch of each dataloader, which can be
processed separately (this was already like this before). Each dataloader batch is
passed to all ACs in the SwitchDAT forward pass. The output of the corresponding AC for
that dataloader is returned to the ensemble model and will be used to compute the
ensemble loss. Its gradients will be backpropagated to the ASR model. All other ACs will
be trained with the same batch within SwitchDAT's forward pass, where the batch is seen
as 'accented'. They are trained with Adam optimizers and the LR defined in the
optimizer config.
"""

import torch
from base import BaseTestClass


class TestSwitchDAT(BaseTestClass):
    def test_loss_change(self):
        """
        The AC loss should increase or decrease, depending on the mode of
        the ACs comprised by the SwitchDAT adversary.
        """
        self.config.ensemble.action = "train"
        self.config.ensemble.mode = "SwitchDAT"
        ac_modes = {"DAT": self.assertGreaterEqual, "MTL": self.assertLessEqual}
        for mode, func in ac_modes.items():
            self.config.ac.mode = mode
            self._init_model()
            for optim_asr in [True, False]:
                with self.subTest(msg=f"{mode} - optim only ASR: {optim_asr}"):
                    return self._loss_change(optim_asr, func)

    def _loss_change(self, optim_asr, func):
        if optim_asr is True:
            self.optim = torch.optim.Adam(self.model.asr.parameters(), 0.01)
        for dl_idx, dataloader in enumerate(self.dataloaders):
            self.model.dataloader_idx = dl_idx
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
                if i == 3:
                    break  # break after three batches of each dl
            if dl_idx == 1:
                break  # break after 2 dls (one standard, one accented)

    def test_ac_training(self):
        """
        Check that the ACs are trained within the forward method of SwitchDAT.
        All ACs that are assigned to an accent different than the current one,
        should be trained with the current batch.
        """
        self.config.ensemble.action = "train"
        self.config.ensemble.mode = "SwitchDAT"
        self.config.ac.mode = "DAT"
        self._init_model()

        for dl_idx, dataloader in enumerate(self.dataloaders):
            self.model.dataloader_idx = dl_idx
            for i, batch in enumerate(dataloader):
                ac_layers = self._get_ac_layers()
                signal, signal_len, _, _ = batch
                self.model.asr.forward(
                    input_signal=signal, input_signal_length=signal_len
                )
                new_ac_layers = self._get_ac_layers()
                for ac_idx, ac in enumerate(ac_layers):
                    if ac_idx == dl_idx:  # AC of this accent should not change
                        self._should_remain(
                            f"DL {dl_idx}, AC {ac_idx}",
                            ac_layers[ac_idx],
                            new_ac_layers[ac_idx],
                        )
                    else:  # AC of other accents should change
                        self._should_change(
                            f"DL {dl_idx}, AC {ac_idx}",
                            ac_layers[ac_idx],
                            new_ac_layers[ac_idx],
                        )
                if i == 3:
                    break  # break after three batches of each dl
            if dl_idx == 1:
                break  # break after 2 dls (one standard, one accented)

    def _get_ac_layers(self):
        """Return the wights of the last layer of each AC."""
        layers = list()
        for key, ac in self.model.ac._modules.items():
            if key not in self.model.seen_langs:
                continue
            layers.append(ac.fc3.layer.weight.data.clone())
        return layers
