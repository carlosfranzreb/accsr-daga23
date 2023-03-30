import logging

import torch
from torch.nn import CrossEntropyLoss
import pytorch_lightning as pl
from nemo.core.optim.lr_scheduler import compute_max_steps

from src.init_adversary import init_adversary
from src.init_asr import init_asr
from src.utils import to_lang_dict, setup_dataloaders


class Model(pl.LightningModule):
    """
    Model that combines an ASR model (ASR) with an accent classifier (AC).

    The init performs the following steps:
    - Initializes the AC (defined in the config).
    - Retrieves the weight of the AC task and the no. of shared ASR layers from the config.
    - Registers the forward hook to the AC on the ASR.
    - Initializes the train and val. dataloaders. The val. DL comprises one for the ASR
        (index 0) and one for the AC (index 1).
    - Loads the accent labels of the train set and the accent mapping (IDs to langs).
    - Initializes the dicts to store sample losses of AC and ASR.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.asr = init_asr(config.asr)
        self.action = config.ensemble.action
        self.mode = config.ensemble.mode
        self.seen_langs = config.data.seen_accents
        self.all_langs = config.data.seen_accents + config.data.unseen_accents
        if "asr" not in self.action:
            self.binary = config.ac.binary
            self.ac_out = None  # output of the accent classifier
            self.branch = config.ensemble.branch
            self.ac_weight = config.ensemble.ac_weight
            self.asr_weight = config.ensemble.asr_weight
            self.ac_loss_fn = CrossEntropyLoss()
            if self.action == "train_ac":
                config.ensemble.mode = "MTL"
            self.ac = init_adversary(config.ac, config.ensemble.mode, self.seen_langs)
            self.dataloader_idx = None  # required by the AC
            self.fwd_hook = self.asr.encoder.layers[self.branch].register_forward_hook(
                self._forward_classifier()
            )
            if self.binary is True:
                self.ac_classes = ["Standard", "Accented"]
            else:
                self.ac_classes = config.data.seen_accents
        self.optimizer = None  # initialized in configure_optimizers
        self.scheduler = None  # initialized in configure_optimizers
        self.asr.setup_optimizer_param_groups = self.setup_optimizer_param_groups

    def configure_optimizers(self):
        """
        Use the same optimizers as the ASR, but change the LR when pre-training
        the AC. Also change the optimizer's parameters depending on which task
        we are performing (accent pre-training or normal training).
        """
        if "sched" not in self.config["optim"]:
            pass
        elif self.config.optim.sched.max_steps is True:
            if self.action == "train_asr":
                train_dl = self.pretrain_dataloader()
            else:
                train_dl = self.train_dataloader()
            n_batches = sum([len(dl) for dl in train_dl])
            self.config.optim.sched.max_steps = compute_max_steps(
                self.config.trainer.max_epochs,
                self.config.trainer.accumulate_grad_batches,
                self.config.trainer.limit_train_batches,
                self.config.data.config.num_workers,
                n_batches * self.config.data.config.batch_size,
                self.config.data.config.batch_size,
                self.config.data.config.drop_last,
            )
        else:
            del self.config.optim.sched.max_steps

        self.optimizer, self.scheduler = self.asr.setup_optimization(self.config.optim)
        if self.scheduler is None:
            return self.optimizer
        return [self.optimizer], [self.scheduler]

    def pretrain_dataloader(self):
        """
        This is not recognized by PTL. We use it to pre-train the ASR model
        on the standard accent, for domain adaptation purposes.
        """
        return setup_dataloaders(self.asr, self.config.data, "pretrain")

    def train_dataloader(self):
        """
        If we are training a binary classifier, merge the training files
        containing accented speech. If this is not done, each training loop
        would comprise more accented data than standard data, as one batch
        is retrieved from each dataset at each time step.
        """
        if self.binary is True:
            return setup_dataloaders(self.asr, self.config.data, "train", 1)
        else:
            return setup_dataloaders(self.asr, self.config.data, "train")

    def val_dataloader(self):
        return setup_dataloaders(self.asr, self.config.data, "test")

    def training_step(self, batch, batch_idx):
        """
        Remove the indices from the batch and use them to obtain the accent labels
        of the samples. Run a training step of the CT, which will also run the forward
        pass of the AC, as it is registered as a hook. Compute the loss of the AC with
        the obtained accent labels, and add both losses together with the appropriate
        weights (defined in the config). If we are currently only pre-training the AC,
        return directly the AC loss.
        """
        if self.action == "train_asr":
            try:
                asr_loss = self.asr.training_step(batch, batch_idx)["loss"]
                self.log("ASR Loss", asr_loss)
                return asr_loss
            except Exception as exc:
                logging.warn(f"Error on batch {batch_idx}: {exc}")
        elif self.action == "train_ac":
            ac_loss_arr = list()
            for dataloader_idx, dl_batch in enumerate(batch):
                class_idx = self.check_binary(dataloader_idx)
                self.asr.forward(
                    input_signal=dl_batch[0], input_signal_length=dl_batch[1]
                )
                ac_loss_arr.append(self._ac_loss(class_idx))
            ac_loss = sum(ac_loss_arr) / len(ac_loss_arr)
            self.log("AC Loss", ac_loss)
            return ac_loss
        elif self.action == "train":
            loss_arr = list()
            for dataloader_idx, dl_batch in enumerate(batch):
                self.dataloader_idx = dataloader_idx
                class_idx = self.check_binary(dataloader_idx)
                asr_loss = self.asr.training_step(dl_batch, batch_idx)["loss"]
                ac_loss = self._ac_loss(class_idx)
                total_loss = self.ac_weight * ac_loss + self.asr_weight * asr_loss
                loss_arr.append(total_loss)
                lang = self.seen_langs[dataloader_idx]
                self.log_dict(
                    {
                        f"Total Loss ({lang})": total_loss,
                        f"ASR Loss ({lang})": asr_loss,
                        f"AC Loss ({lang})": ac_loss,
                    }
                )
            loss = sum(loss_arr) / len(loss_arr)
            self.log("Total Loss", loss)
            return loss
        else:
            raise ValueError(
                f"Action {self.action} is invalid. Change model.action in the config."
            )

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        """
        If we are pre-training the AC, run only the encoder of the ASR. Otherwise,
        run its validation step, which outputs WER. With the AC predictions:
        1. Log the AC Loss for the batch.
        2. If only one accent is present in the batch, log the AC Loss mentioning
        the accent.
        3. Update the AC's confusion matrix.
        """
        self.dataloader_idx = dataloader_idx
        outputs = dict()
        if self.action == "train_ac":  # don't validate the ASR model
            self.asr.forward(input_signal=batch[0], input_signal_length=batch[1])
        else:
            asr_logs = self.asr.validation_step(batch, batch_idx)
            outputs["val_wer"] = asr_logs["val_wer"]
        if (
            self.action == "train_asr"
            or (self.binary is False or self.mode == "SwitchDAT")
            and dataloader_idx >= len(self.seen_langs)
        ):
            return outputs  # either the whole AC or this accent is not relevant
        outputs["val_ac_loss"] = self._ac_loss(self.check_binary(dataloader_idx))
        outputs["val_ac_predictions"] = self.ac_out.argmax(dim=1)
        return outputs

    def validation_epoch_end(self, outputs):
        all_outs = self.all_gather(outputs)
        if isinstance(all_outs[0], dict):  # true when there is only one dataloader
            all_outs = [all_outs]  # update struct to conform to the rest of the code
        if "asr" not in self.action:
            self.log_ac_predictions(all_outs)
            self.log_averages(all_outs, "val_ac_loss")
        if "ac" not in self.action:
            self.log_averages(all_outs, "val_wer")

    def _ac_loss(self, class_idx):
        """Compute the AC Loss with the given class as target for the whole batch."""
        class_indices = (
            torch.ones(self.ac_out.shape[0], dtype=torch.int64, device=self.device)
            * class_idx
        )
        return self.ac_loss_fn(self.ac_out, class_indices)

    def log_ac_predictions(self, outputs):
        """
        Log the prediction statistics accumulated throughout the validation epoch.
        It logs the number of predictions made for each language, as well as the
        no. of predictions per language, grouped by correct language. Also log
        precision, recall and F1-score, for each language and overall.
        """
        predictions = torch.zeros(
            len(self.ac_classes), len(self.ac_classes), device=self.device
        )
        for dl_idx, outs in enumerate(outputs):
            if (self.binary is False or self.mode == "SwitchDAT") and dl_idx >= len(
                self.seen_langs
            ):
                continue  # no AC predictions for this accent
            class_idx = self.check_binary(dl_idx)
            ac_preds = torch.cat([out["val_ac_predictions"].flatten() for out in outs])
            for cls in range(len(self.ac_classes)):
                predictions[class_idx, cls] += torch.sum(ac_preds == cls)
        n_preds = predictions.sum(dim=0)  # no. of times a lang was predicted
        self.logger.experiment.add_scalars(
            "No. of times each accent was predicted",
            to_lang_dict(n_preds, self.ac_classes),
            global_step=self.current_epoch,
        )
        for i, lang in enumerate(self.ac_classes):
            self.logger.experiment.add_scalars(
                f"No. of times each accent was predicted when the correct accent is {lang}",
                to_lang_dict(predictions[i], self.ac_classes),
                global_step=self.current_epoch,
            )
        true_pos = predictions.diagonal()
        no_diag = predictions * ~torch.eye(
            predictions.shape[0], dtype=bool, device=self.device
        )
        false_pos = no_diag.sum(dim=0)
        false_neg = no_diag.sum(dim=1)
        precisions = true_pos / (true_pos + false_pos)
        recalls = true_pos / (true_pos + false_neg)
        f1scores = (2 * precisions * recalls) / (precisions + recalls)
        for title, data in {
            "Precision": precisions,
            "Recall": recalls,
            "F1-score": f1scores,
        }.items():
            data_dict = to_lang_dict(data, self.ac_classes)
            data_dict["avg"] = data @ n_preds / n_preds.sum()
            self.logger.experiment.add_scalars(
                title, data_dict, global_step=self.current_epoch
            )

    def log_averages(self, outputs, metric):
        """
        Given a list of lists containing multiple data points per list, compute the
        average of each list, as well as the unweighted average across lists. Log the
        averages with the given title, combined with the langs of the list. The mapping
        of data to langs is done through their indices in their respective lists. Log
        the unweighted average separately so it can be used for checkpointing. This method
        should only be called at the end of every epoch.
        """
        avgs = dict()
        for dl_idx, outs in enumerate(outputs):
            if (
                metric == "val_ac_loss"
                and (self.binary is False or self.mode == "SwitchDAT")
                and dl_idx >= len(self.seen_langs)
            ):
                continue  # no AC predictions for this accent
            values = [out[metric].mean().item() for out in outs]
            avgs[self.all_langs[dl_idx]] = sum(values) / len(values)
        avgs["avg (unweighted)"] = torch.tensor(list(avgs.values())).mean().item()
        self.log(
            f"{metric} - unweighted avg.",
            avgs["avg (unweighted)"],
        )
        self.logger.experiment.add_scalars(
            f"{metric} per correct accent",
            avgs,
            global_step=self.current_epoch,
        )

    def check_binary(self, dataloader_idx):
        """
        - If we are not training a binary AC, return the dataloader_idx unchanged.
        - If we are training a binary AC, return the binary class associated to the
            given dataloader_idx (standard or accented). When the AC is binary,
            datasets whose dataloader_idx is other than the first (i.e. 0) belong to
            the second class, i.e. 1.
        - When we are training a Switch DAT (many classifiers), we always return
            zero, as the AC used for this batch always has this dataloader_idx as
            standard.
        """
        if self.mode == "SwitchDAT":
            return 0
        elif self.binary is True:
            return int(dataloader_idx > 0)
        return dataloader_idx

    def _forward_classifier(self):
        """
        Runs a forward pass of the classifier. Called by the forward hook
        registered on the conformer encoder.
        """

        def hook(model, input, output):
            self.ac_out = self.ac(output, self.dataloader_idx)

        return hook

    def setup_optimizer_param_groups(self):
        """
        Define separate parameter groups for the AC and the ASR, each with its
        own optimizer args. This method overwrites the default behaviour of the
        NeMo base model.
        """
        args = {"ac_args": {}, "asr_args": {}}
        for key in args:
            if key in self.config.optim:
                args[key] = self.config.optim[key]
                del self.config.optim[key]
        if "asr" in self.action:
            self.asr._optimizer_param_groups = [{"params": self.parameters()}]
        elif "ac" in self.action:
            self.asr._optimizer_param_groups = [{"params": self.ac.parameters()}]
        else:
            self.asr._optimizer_param_groups = [
                {"params": self.ac.parameters(), **args["ac_args"]},
                {"params": self.asr.parameters(), **args["asr_args"]},
            ]
