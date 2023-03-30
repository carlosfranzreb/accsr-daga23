import sys
import torch
from nemo.collections.asr.models.rnnt_bpe_models import EncDecRNNTBPEModel


def init_asr(config):
    """Return an initialized ASR model as defined in the config."""
    model_class = getattr(sys.modules[__name__], config["classname"])
    model = model_class.from_pretrained(config["pretrained"])
    if config["ckpt"] is not None:
        ckpt = torch.load(config["ckpt"], map_location=torch.device("cpu"))
        ckpt_dict = {k[4:]: v for k, v in ckpt["state_dict"].items() if k.startswith("asr.")}
        model_dict = model.state_dict()
        model_dict.update(ckpt_dict)
        model.load_state_dict(model_dict)
    if torch.cuda.is_available():
        model.cuda()
    return model
