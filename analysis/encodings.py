"""Plot the ASR encoder representations with t-SNE for each validation sample,
colored by accent. Store the config, results and plot in `logs/analysis/encoder_viz`,
with version folders (just like PTL does)."""


import os
import json

import torch
import torch.nn.functional as F
from sklearn.manifold import TSNE
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
from omegaconf import OmegaConf
from scipy.spatial.distance import cdist
from scipy.stats import wasserstein_distance

from src.model import Model
from src.utils import get_device


class EncoderViz:
    def __init__(self, config, trainer):
        self.log_dir = trainer.logger.log_dir
        self.n_samples = config.components.EncoderViz.n_samples
        self.device = get_device(trainer)
        config.ensemble = OmegaConf.create(
            {
                "branch": 3,  # this doesn't matter
                "weight": 0.5,  # this doesn't matter
                "mode": "DAT",  # this doesn't matter
                "action": "evaluate_asr",  # so the AC is not initialized
            }
        )
        config.data.config_test.shuffle = False  # for reproducibility
        config.asr.ckpt = config.get("ckpt_asr", None)
        model = Model(config)
        self.encoder = model.asr.encoder
        self.preprocessor = model.asr.preprocessor.to(self.device)
        self.encoder.eval()
        self.encoder = self.encoder.to(self.device)
        self.dataloaders = model.val_dataloader()
        self.optim = model.configure_optimizers()
        if isinstance(self.optim, tuple):
            self.optim = self.optim[0]
        self.n_components = 2
        if "tsne" in config.components.EncoderViz:
            self.tsne = TSNE(**config.components.EncoderViz.tsne)
            if "n_components" in config.components.EncoderViz.tsne:
                self.n_components = config.components.EncoderViz.tsne.n_components
        else:
            self.tsne = TSNE()
        self.langs = config.data.seen_accents + config.data.unseen_accents
        json.dump(list(self.langs), open(os.path.join(self.log_dir, "langs.json"), "w"))

    def run(self):
        """For each pair of branch/ckpt, check the gradients of the ASR and
        the AC that are backpropagated to the branching layer."""
        dump_tsne = os.path.join(self.log_dir, "tsne")
        encoder_out_dir = os.path.join(self.log_dir, "enc_out")
        if os.path.exists(dump_tsne):
            raise RuntimeError("Dump folder already exists. It shouldn't.")
        if not os.path.exists(encoder_out_dir):
            os.makedirs(encoder_out_dir)
            self.compute_representations(encoder_out_dir)
        for branch in os.listdir(encoder_out_dir):
            encodings, labels = self.load_encodings(
                os.path.join(encoder_out_dir, branch)
            )
            encoding_avgs = encodings.mean(axis=2)
            embeddings = self.tsne.fit_transform(encoding_avgs)
            tsne_folder = os.path.join(dump_tsne, branch)
            os.makedirs(tsne_folder, exist_ok=True)
            np.save(os.path.join(tsne_folder, "vecs.npy"), embeddings)
            np.save(os.path.join(tsne_folder, "labels.npy"), labels)
            if self.n_components == 2:
                self.plot_embeddings(embeddings, labels, tsne_folder)
            compute_dists(embeddings, labels, tsne_folder)

    def compute_representations(self, out_dir):
        """Compute the encoder representations for all samples of all datasets.
        Store them as tensors in the given folder, one per dataset."""
        for branch in range(len(self.encoder.layers) - 1):
            branch_str = f"b{branch}"
            self.encoder.layers[branch].register_forward_hook(
                self.intermediate_hook(os.path.join(out_dir, branch_str))
            )
            os.makedirs(os.path.join(out_dir, branch_str), exist_ok=True)
        last_branch = f"b{len(self.encoder.layers) - 1}"
        last_dumpdir = os.path.join(out_dir, last_branch)
        os.makedirs(os.path.join(out_dir, last_branch), exist_ok=True)
        for i, dl in enumerate(self.dataloaders):
            self.dataloader_idx = i
            self.encodings = {
                f"b{b}": None for b in range(len(self.encoder.layers) - 1)
            }
            self.encodings[last_branch] = None
            samples_done = 0
            for wav, wav_len, _, _ in dl:
                if wav.shape[0] + samples_done > self.n_samples:
                    wav = wav[: self.n_samples - samples_done]
                    wav_len = wav_len[: self.n_samples - samples_done]
                samples_done += wav.shape[0]
                with torch.no_grad():
                    processed, processed_len = self.preprocessor(
                        input_signal=wav.to(self.device),
                        length=wav_len.to(self.device),
                    )
                    enc_out = (
                        self.encoder(
                            audio_signal=processed,
                            length=processed_len,
                        )[0]
                        .detach()
                        .cpu()
                    )
                self.store_output(enc_out, last_dumpdir)
                if samples_done == self.n_samples:
                    break

    def load_encodings(self, folder):
        """Load and concatenate all encodings found in the given folder and
        return them as a numpy tensor. Return an additional array with the accent
        labels of the samples."""
        encodings = None
        labels = list()
        for dl_dir in os.listdir(folder):
            for fname in os.listdir(os.path.join(folder, dl_dir)):
                data = torch.load(os.path.join(folder, dl_dir, fname))
                encodings = concat(encodings, data)
                labels.append(torch.ones(data.shape[0]) * int(dl_dir[11:]))
        return encodings.detach().numpy(), torch.cat(labels).numpy()

    def intermediate_hook(self, dump_folder):
        """Return a hook that stores the output of the current layer in the
        `encodings` attribute. `dump_folder`consists of the folder where the encoder
        outputs are stored and the folder for this branch."""

        def hook(module, input, output):
            self.store_output(output.detach().cpu().transpose(1, 2), dump_folder)

        return hook

    def store_output(self, tensor, dump_folder):
        """Dump the given tensor in the folder, named with a number that equals
        the current number of files in the folder. Add the dataloader index to
        the folder name."""
        dump_folder = os.path.join(dump_folder, f"dataloader_{self.dataloader_idx}")
        os.makedirs(dump_folder, exist_ok=True)
        fname = len(os.listdir(dump_folder))
        torch.save(tensor, os.path.join(dump_folder, f"{fname}.pt"))

    def plot_embeddings(self, vecs, labels, dump_folder):
        """Plot the t-SNE vecs and save them to the given file. Store also
        the cosine and euclidean distances between all pairs of means of the accents."""
        n_accents = int(np.max(labels)) + 1
        colors = cm.rainbow(np.linspace(0, 1, n_accents))
        avgs = np.zeros((n_accents, 2))
        for i in range(n_accents):
            accent_data = vecs[np.argwhere(labels == i)].squeeze()
            mean = np.mean(accent_data, axis=0)
            avgs[i] = mean
            plt.scatter(
                accent_data[:, 0],
                accent_data[:, 1],
                color=colors[i],
                label=self.langs[i],
                alpha=0.2,
            )
            plt.scatter(
                mean[0], mean[1], facecolors=colors[i], edgecolors="black", s=100
            )
        plt.legend()
        plt.savefig(os.path.join(dump_folder, "plot.png"))
        plt.close()


def compute_dists(vecs, labels, dump_folder):
    """
    Compute and store the following distances:
    - cosine and euclidean distances between all pairs of means of the accents
    - earth mover's distance between the distributions of the accents
    """
    # sort the vecs together with the labels
    idx = np.argsort(labels)
    labels = labels[idx]
    vecs = vecs[idx]
    # reshape the vecs so that the vecs of each label are together
    n_labels = labels.unique().shape[0]
    samples_per_label = labels.shape[0] // n_labels
    vecs = vecs.reshape(n_labels,samples_per_label, vecs.shape[1])
    # compute the mean embedding for each label
    means = vecs.mean(axis=1)
    for metric in ["cosine", "euclidean"]:
        dists = cdist(means, means, metric=metric)
        json.dump(
            dists.tolist(),
            open(os.path.join(dump_folder, f"{metric}_dists.json"), "w"),
        )
    # Compute the earth mover's distance between the distributions
    emd_matrix = np.zeros((vecs.shape[0], vecs.shape[0]))
    for i in range(vecs.shape[0]):
        for j in range(i, vecs.shape[0]):
            emd = wasserstein_distance(vecs[i].reshape(-1), vecs[j].reshape(-1))
            emd_matrix[i][j] = emd
            emd_matrix[j][i] = emd
    # dump the EMD matrix to a numpy file
    np.save(os.path.join(dump_folder, "emd.npy"), emd_matrix)


def concat(tensor_a, tensor_b):
    """Given two tensors with shapes (B x D x T), pad the shorter one and return
    their concatenation. If one of the tensors is null, return the other."""
    if tensor_a is None:
        return tensor_b
    elif tensor_b is None:
        return tensor_a
    elif tensor_a.shape[2] < tensor_b.shape[2]:
        tensor_a = pad_dim0(tensor_a, tensor_b.shape[2] - tensor_a.shape[2])
    elif tensor_a.shape[2] > tensor_b.shape[2]:
        tensor_b = pad_dim0(tensor_b, tensor_a.shape[2] - tensor_b.shape[2])
    return torch.cat((tensor_a, tensor_b))


def pad_dim0(data, pad_size):
    """Pad-right the given tensor on its first dimension."""
    return F.pad(data, (0, pad_size, 0, 0, 0, 0), value=0)


if __name__ == "__main__":
    print(os.getcwd())
    log_dir = "logs/de/analysis/version_17"
    tsne_dir = os.path.join(log_dir, "tsne", "b13")
    vecs = np.load(os.path.join(tsne_dir, "vecs.npy"))
    labels = np.load(os.path.join(tsne_dir, "labels.npy"))
    compute_dists(vecs, labels, tsne_dir)
