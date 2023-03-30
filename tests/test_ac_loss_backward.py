"""Test whether the AC Loss backpropagates to the ASR model."""


from base import BaseTestClass


class TestAcLossBwd(BaseTestClass):
    def test_ac_loss_backward(self):
        """Use the 'train ensemble' mode to add all AC and ASR parameters to the
        optimizer, as in 'train AC' the ASR model is frozen. Still, we only
        backpropagate the classifier loss, and expect the ASR encoder components
        before the branching out to change."""
        self.config.ensemble.action = "train"
        self._init_model()
        params = [np for np in self.model.named_parameters() if np[1].requires_grad]
        before, after = self._train_step(params)
        for param_name, val_before in before.items():
            if "asr.encoder.layers" in param_name:
                layer_nr = self._get_enc_layer(param_name)
                if layer_nr > self.config["ensemble"]["branch"]:
                    self._should_remain(param_name, val_before, after[param_name])
                    continue
            elif "asr.decoder" in param_name or "asr.joint" in param_name:
                self._should_remain(param_name, val_before, after[param_name])
                continue
            self._should_change(param_name, val_before, after[param_name])

    def _train_step(self, params):
        before = {name: p.clone() for (name, p) in params}
        for i, batch in enumerate(self.dataloaders[0]):
            signal, signal_len, _, _ = batch
            self.model.asr.forward(input_signal=signal, input_signal_length=signal_len)
            loss = self.model._ac_loss(0)
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()
            break
        after = {name: p.clone() for (name, p) in params}
        return before, after

    def _get_enc_layer(self, param_name):
        """We assume that the parameter name starts with 'asr.encoder.layers.'
        and start looking for numbers from the on."""
        layer_nr = ""
        for i in range(19, len(param_name)):
            if param_name[i] == ".":
                try:
                    return int(layer_nr)
                except Exception:
                    print(param_name)
            layer_nr += param_name[i]
