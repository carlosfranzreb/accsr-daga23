import pytorch_lightning as pl
from nemo.core.optim.lr_scheduler import compute_max_steps
from torch.utils.data import TensorDataset, DataLoader
import torch


class TestLR(pl.LightningModule):
    """
    Empty model to test the LR scheduler. The training step only logs the current LR
    and returns null. The dataloader is a dummy one to avoid all the processing done
    by NeMo's dataloader. It configures the optimizer and scheduler as the original model.
    """

    def __init__(self, ct, config):
        super().__init__()
        self.ct = ct
        self.config = config
        data_config = config["data"]["train_ds"]
        n_samples = len(open(data_config["manifest_filepath"]).readlines())
        self.train_dl = DataLoader(
            TensorDataset(torch.empty(n_samples)),
            batch_size=data_config["batch_size"],
            drop_last=data_config["drop_last"],
            num_workers=data_config["num_workers"],
        )
        self.optimizer = None  # initialized in configure_optimizers
        self.scheduler = None  # initialized in configure_optimizers

    def configure_optimizers(self):
        """Configure the optimizers as defined in the config file. Use the
        setup function of the conformer transducer."""
        if "sched" not in self.config["optim"]:
            pass
        elif self.config["optim"]["sched"]["max_steps"] is True:
            self.config["optim"]["sched"]["max_steps"] = compute_max_steps(
                self.config["trainer"]["max_epochs"],
                self.config["trainer"]["accumulate_grad_batches"],
                self.config["trainer"]["limit_train_batches"],
                self.config["data"]["train_ds"]["num_workers"],
                len(self.train_dl) * self.config["data"]["train_ds"]["batch_size"],
                self.config["data"]["train_ds"]["batch_size"],
                self.config["data"]["train_ds"]["drop_last"],
            )
        else:
            del self.config["optim"]["sched"]["max_steps"]
        self.optimizer, self.scheduler = self.ct.setup_optimization(
            self.config["optim"]
        )
        self.ct = None
        if self.scheduler is None:
            return self.optimizer
        return [self.optimizer], [self.scheduler]

    def training_step(self, batch, batch_idx):
        """Log the LR of the current training step."""
        self.log("lr", self.optimizer.param_groups[0]["lr"])
        return None
