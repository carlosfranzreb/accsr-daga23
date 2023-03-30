"""Test whether the AC Loss backpropagates to the ASR model."""


import unittest
import os
import shutil

import torch
from omegaconf import OmegaConf

from src.init_experiment import init_exp


class BaseTestClass(unittest.TestCase):
    def setUp(self):
        self.config = OmegaConf.create(
            {
                "job": "test",
                "language": "de",
                "ensemble": {
                    "branch": 3,
                    "ac_weight": 0.5,
                    "asr_weight": 0.5,
                    "mode": "DAT",
                    "action": None,  # this value is set within each test
                },
                "asr": {
                    "classname": "EncDecRNNTBPEModel",
                    "pretrained": "stt_de_conformer_transducer_large",
                    "ckpt": None,
                },
                "ac": {
                    "classname": "AC",
                    "ckpt": None,
                    "n_accents": 2,
                    "dropout": None,
                    "binary": True,
                    "optim": {"lr": 0.01},
                },
                "data": {
                    "root": "/Users/cafr02/datasets",
                    "folder": "data/debug",
                    "seen_accents": ["de", "at"],
                    "unseen_accents": ["ca"],
                    "config": {
                        "sample_rate": 16000,
                        "batch_size": 1,
                        "num_workers": 0,
                        "pin_memory": True,
                        "use_start_end_token": False,
                        "trim_silence": False,
                        "max_duration": 18.9,
                        "min_duration": 0.1,
                        "drop_last": False,
                    },
                    "config_train": {"shuffle": False},
                    "config_test": {"shuffle": False},
                },
                "optim": {
                    "name": "adamw",
                    "lr": 0.1,
                    "betas": [0.9, 0.98],
                    "weight_decay": 0.001,
                },
                "trainer": {
                    "max_epochs": 20,
                    "accumulate_grad_batches": 1,
                    "limit_train_batches": 1.0,
                    "limit_val_batches": 1.0,
                },
                "checkpointing": {
                    "monitor": "val_wer - unweighted avg.",
                    "mode": "min",
                },
            }
        )
        self.config_file = "tests/exp_folder/config.yaml"

    def tearDown(self):
        """Remove the config file and the log folder."""
        if os.path.exists(self.config_file):
            os.remove(self.config_file)
        shutil.rmtree(self.trainer.logger.log_dir)

    def _init_model(self):
        """Use the class config to init model, dataloader and optimizer."""
        OmegaConf.save(self.config, self.config_file)
        args = OmegaConf.create(
            {
                "config": self.config_file,
                "devices": 1,
                "debug": False,
                "accelerator": "cpu",
            }
        )
        self.config, self.trainer, self.model = init_exp(args)
        self.dataloaders = self.model.val_dataloader()
        self.optim = self.model.configure_optimizers()

    def _should_remain(self, param_name, val_before, val_after):
        """Assert that the value of the given parameter does not change."""
        with self.subTest(msg=f"{param_name} does not change"):
            self.assertTrue(
                (val_before == val_after).all(),
                msg=f"{param_name} changes; it should not",
            )

    def _should_change(self, param_name, val_before, val_after):
        """Assert that the value of the given parameter changes."""
        change = torch.round(torch.sum(torch.abs(val_after - val_before)), decimals=2)
        with self.subTest(msg=f"{param_name} changes ({change})"):
            self.assertTrue(
                (val_before != val_after).any(),
                msg=f"{param_name} does not change; it should",
            )
