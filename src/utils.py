import json
import os

import torch
from omegaconf import OmegaConf, DictConfig
from pytorch_lightning.accelerators import CPUAccelerator, GPUAccelerator


def to_lang_dict(data, langs):
    """Convert the given data (array or tensor) to a dictionary
    with the given languages as keys."""
    if torch.is_tensor(data):
        data = data.tolist()
    if len(data) != len(langs):
        raise ValueError("data and langs must be the same size")
    return {lang: data[i] for i, lang in enumerate(langs)}


def setup_dataloaders(asr_model, config, mode, concat=None):
    """Return a list of  "mode" dataloaders as defined in the config. "mode" can be
    "pretrain", "train" or "test". "train" manifest files are those named "train_{lang}",
    where "lang" is any of the "seen" languages. Similarly, "test" manifest files are
    those named "test_{lang}, where "lang" is any of both seen and unseen languages".
    The config for the dataloaders are created by merging the general configuration
    (config["config"] with that that is specific to the mode (config["config_train"]
    or config["config_test"]. The resulting config contains the data required by the
    NeMo "asr_model", which is used to create the dataloaders. If concat is an int,
    the files corresponding to langs indexed "int" or higher are concatenated and
    returned as a single dataloader. This is useful to create a single training
    dataloader for all accents, when training a binary accent classifier."""
    dl_config = config.config
    dl_config.update(config[f"config_{mode}"])
    if mode == "pretrain":
        dl_config.manifest_filepath = config.pretrain_files
        return asr_model._setup_dataloader_from_config(dl_config)
    dataloaders = []
    if concat is not None:
        dl_config.manifest_filepath = config[f"{mode}_files"][:concat]
        dataloaders.append(asr_model._setup_dataloader_from_config(dl_config))
        dl_config.manifest_filepath = config[f"{mode}_files"][concat:]
        dataloaders.append(asr_model._setup_dataloader_from_config(dl_config))
    else:
        for filepath in config[f"{mode}_files"]:
            dl_config.manifest_filepath = filepath
            dataloaders.append(asr_model._setup_dataloader_from_config(dl_config))
    return dataloaders


def exp_folder(config):
    """Return the experiment folder given the config."""
    folder = config.language
    if config.job == "analysis":
        return os.path.join(folder, "analysis")
    elif config.job == "test":
        return os.path.join(folder, "tests", "exp_folder")
    action = config.ensemble.action
    if "_" in action:
        mode, model = action.split("_")
    else:
        model = "ensemble"
        mode = action
    folder = os.path.join(folder, model, mode)
    if model != "asr":
        ac_type = "binary" if config.ac.binary else "multi-class"
        folder += f"/{ac_type}/b{config.ensemble.branch}"
    if model == "ensemble":
        folder += f"/{config.ensemble.mode}"
    return folder


def create_datafiles(config, mode, log_dir):
    """Add the root folder to the paths of the audiofiles and store the resulting
    manifest files within the experiment folder. These will be deleted once the
    experiment has finished."""
    test = [f"test_{acc}.txt" for acc in config.seen_accents + config.unseen_accents]
    pretrain, train = list(), list()
    if mode == "train_asr":
        pretrain.append(f"pretrain_{config.seen_accents[0]}.txt")
    elif "train" in mode:
        train = [f"train_{acc}.txt" for acc in config.seen_accents]
    for files in [pretrain, train, test]:
        new_paths = list()
        for f in files:
            datafile = os.path.join(config.folder, f)
            new_paths.append(os.path.join(log_dir, datafile))
            if os.path.exists(new_paths[-1]):  # manifest already modified
                continue
            os.makedirs(os.path.dirname(new_paths[-1]), exist_ok=True)
            with open(new_paths[-1], "w") as writer:
                with open(datafile) as reader:
                    for line in reader:
                        obj = json.loads(line)
                        obj["audio_filepath"] = obj["audio_filepath"].replace(
                            "{root}", config.root
                        )
                        writer.write(json.dumps(obj) + "\n")
        if len(new_paths) > 0:
            config[f"{f.split('_')[0]}_files"] = new_paths
    return config


def load_subconfigs(config):
    """Given a config, load all the subconfigs that are specified in the config into
    the same level as the parameter. Configs are specified by parameters ending with
    '_file'. If a value of the config is a dict, call this function again recursively."""
    for key, value in config.items():
        if isinstance(value, DictConfig):
            config[key] = load_subconfigs(value)
        elif key.endswith("_file"):
            new_config = OmegaConf.load(value)
            for key, value in new_config.items():
                config[key] = value
    return config


def get_device(trainer):
    """Return the device used by the trainer."""
    if isinstance(trainer.accelerator, CPUAccelerator):
        return torch.device("cpu")
    elif isinstance(trainer.accelerator, GPUAccelerator):
        return torch.device("cuda")
    else:
        raise RuntimeError("Unknown accelerator type")
