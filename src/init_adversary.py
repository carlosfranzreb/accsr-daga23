from src.switch_dat import SwitchDAT
from src.init_classifier import init_classifier


def init_adversary(config, mode, seen_langs=None):
    """Return an initialized accent classifier as defined in the config. If `mode` is
    "SwitchDAT", return a SwitchDAT object, which requires the `seen_langs` kwarg to
    be a list of the seen accents during training, to create one AC per accent.
    Otherwise the kwarg `sen_langs` is ignored."""
    if mode == "SwitchDAT":
        return SwitchDAT(config, seen_langs)
    else:
        return init_classifier(config, mode, 0)  # default standard accent is 0
