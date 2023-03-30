"""Test whether variables change through the backward pass."""


from base import BaseTestClass


class TestVarChange(BaseTestClass):
    def test_train_ensemble(self):
        """Train the ASR model and the classifier jointly. All parameters should
        require grad and change after every training step."""
        self.config.ensemble.action = "train"
        self._init_model()
        params = [np for np in self.model.named_parameters() if np[1].requires_grad]
        before, after = self._train_step(params, True)
        for name, val_before in before.items():
            self._should_change(name, val_before, after[name])

    def test_train_ac(self):
        """Test classifier pre-training. The ASR model is frozen during this mode and its
        params should not change. Only the classifier parameters should change."""
        self.config.ensemble.action = "train_ac"
        self._init_model()
        params = [np for np in self.model.named_parameters() if np[1].requires_grad]
        before, after = self._train_step(params, True)
        for name, val_before in before.items():
            if "asr" in name:
                self._should_remain(name, val_before, after[name])
                continue
            self._should_change(name, val_before, after[name])

    def test_train_asr(self):
        """Test ASR pre-training. The classifier is not appended to the ASR model
        in this mode and its params should not be present in the optimizer. All
        ASR params should change."""
        self.config.ensemble.action = "train_asr"
        self._init_model()
        params = list()
        for (name, param) in self.model.named_parameters():
            self.assertTrue("ac." not in name, msg=f"{name} should not be in the model")
            if param.requires_grad:
                params.append((name, param))
        before, after = self._train_step(params, False)
        for name, val_before in before.items():
            self._should_change(name, val_before, after[name])

    def _train_step(self, params, wrap_batch):
        """Perform one training step and return the named parameters as a dict,
        both before and after the training step. wrap_batch determines if the
        batch is placed in a list before being passed to the model."""
        before = {name: p.clone() for (name, p) in params}
        for i, batch in enumerate(self.dataloaders[0]):
            if wrap_batch is True:
                batch = [batch]
            loss = self.model.training_step(batch, i)
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()
            break
        after = {name: p.clone() for (name, p) in params}
        return before, after
