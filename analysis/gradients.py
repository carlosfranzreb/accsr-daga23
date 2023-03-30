"""Compare the size of the ASR and AC gradients that arrive at the branching point
for all combinations of the given values for the branch, weights and checkpoints
(both for the AC and the ASR). For each combination, initialize a model and store
the gradients of one batch of each dataloader, both for the ASR and the AC. Compute
also the mean gradient vector for each model, and also their difference in magnitude."""


import os
import json

import torch
from omegaconf import OmegaConf

from src.model import Model
from src.utils import get_device


class GradChecker:
    def __init__(self, config, trainer):
        self.config = config
        self.params = config.components.GradChecker
        self.trainer = trainer
        self.config.data.config_test.shuffle = True
        self.config.ac.ckpt = None
        self.config.asr.ckpt = None
        self.config.ensemble = OmegaConf.create(
            {
                "branch": None,
                "weight": None,
                "mode": "DAT",  # this doesn't matter
                "action": "train",  # so the AC is initialized
            }
        )
        self.device = get_device(trainer)

    def _init_model(
        self, branch, ac_weight, asr_weight, asr_trained=False, ac_trained=False
    ):
        """Use the class config to init model, dataloader and optimizer. asr_trained
        and ac_trained are booleans defining whether a ckpt is used or not for each
        model"""
        self.config.ensemble.branch = branch
        self.config.ensemble.ac_weight = ac_weight
        self.config.ensemble.asr_weight = asr_weight
        if asr_trained is True:
            self.config.asr.ckpt = self.config.ckpt_asr
        if ac_trained is True:
            self.config.ac.ckpt = self.config.ckpt_ac[f"b{branch}"]
        self.model = Model(self.config)
        self.model.to(self.device)
        self.dataloaders = self.model.val_dataloader()
        self.optim = self.model.configure_optimizers()
        if isinstance(self.optim, tuple):
            self.optim = self.optim[0]

    def run(self):
        """For each pair of branch/ckpt, check the gradients of the ASR and the AC
        that are backpropagated to the branching layer. Dump them, along with the
        avg. across gradients, in a json file. Dump also the avg. across batches
        and their absolute difference in magnitude."""
        dump_file = os.path.join(self.trainer.logger.log_dir, "gradients.json")
        if os.path.exists(dump_file):
            raise RuntimeError("Dump file already exists. It shouldn't.")
        out = list()
        for branch in self.params.branches:
            for ac_weight in self.params.ac_weights:
                for asr_weight in self.params.asr_weights:
                    for trained_asr in self.params.trained_asr:
                        for trained_ac in self.params.trained_ac:
                            self._init_model(
                                branch, ac_weight, asr_weight, trained_asr, trained_ac
                            )
                            grads = self.check_grads(branch, ac_weight)
                            grads_ac = torch.cat([g["ac"].unsqueeze(0) for g in grads])
                            grads_asr = torch.cat(
                                [g["asr"].unsqueeze(0) for g in grads]
                            )
                            grads_ac_avg = grads_ac.mean(dim=0)
                            grads_asr_avg = grads_asr.mean(dim=0)
                            grads_div = grads_ac_avg.abs() / grads_asr_avg.abs()
                            out.append(
                                {
                                    "branch": branch,
                                    "ac_weight": ac_weight,
                                    "trained_asr": trained_asr,
                                    "trained_ac": trained_ac,
                                    "grads_ac": grads_ac.tolist(),
                                    "grads_asr": grads_asr.tolist(),
                                    "grads_ac_avg": grads_ac.mean(dim=0).tolist(),
                                    "grads_asr_avg": grads_asr.mean(dim=0).tolist(),
                                    "grads_div": grads_div.tolist(),
                                    "grads_div_avg": grads_div.mean().item(),
                                }
                            )
        json.dump(out, open(dump_file, "w"))

    def check_grads(self, branch, ac_weight):
        """Backpropagate the ASR loss and the AC loss separately and never change the
        parameters. Do so with 1 batch of each validation set."""
        grads = list()
        for dl in self.dataloaders:
            batch = next(iter(dl))
            batch = (b.to(self.device) for b in batch)
            asr_loss = self.model.asr.training_step(batch, 0)["loss"] * (1 - ac_weight)
            ac_loss = self.model._ac_loss(0) * ac_weight
            grads.append(
                {
                    "ac": self.get_grad(ac_loss, branch),
                    "asr": self.get_grad(asr_loss, branch),
                }
            )
        return grads

    def get_grad(self, loss, branch):
        """Compute the backward pass of the given loss and return the gradient of the
        last layer of the `branch`'th conformer block."""
        self.optim.zero_grad()
        loss.backward(retain_graph=True)
        return (
            self.model.asr.encoder.layers[branch].norm_out.weight.grad.detach().clone()
        )
