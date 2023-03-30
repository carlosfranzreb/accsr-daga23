import sys
import torch

from src.accent_classifiers.classifier import AC


def init_classifier(config, mode, standard):
    """Return an initialized accent classifier as defined in the config. `standard` is
    the standard accent for OneWayDAT, where accents other than `standard`` are
    reversed; `standard` is not. DAT mode ignores this `standard` arg and reverses all
    gradients."""
    model_class = getattr(sys.modules[__name__], config.classname)
    model = model_class(config.n_accents, config.dropout, mode, standard)
    if config.ckpt is not None:
        ckpt = torch.load(config.ckpt, map_location=torch.device("cpu"))
        ckpt_dict = {k[3:]: v for k, v in ckpt["state_dict"].items() if k[:3] == "ac."}
        model_dict = model.state_dict()
        model_dict.update(ckpt_dict)
        model.load_state_dict(model_dict)
    if torch.cuda.is_available():
        model.cuda()
    return model
