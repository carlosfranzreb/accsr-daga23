"""Analyse results with the analysis scripts stored in the `analysis` folder."""


import sys
import os
from argparse import ArgumentParser
from shutil import rmtree

from omegaconf import OmegaConf

from analysis.gradients import GradChecker
from analysis.encodings import EncoderViz
from analysis.mapsswe import MAPSSWE
from analysis.errors import ErrorCounter
from src.init_experiment import init_exp


def main(args):
    config, trainer, _ = init_exp(args)
    for classname in config.components:
        cls = getattr(sys.modules[__name__], classname)
        obj = cls(config, trainer)
        obj.run()
    rmtree(os.path.join(trainer.logger.log_dir, "data"))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", default="config/analysis.yaml")
    parser.add_argument("--accelerator", default="gpu")
    parser.add_argument("--devices", default="1")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    main(args)
