from typing import Optional, Union

import torch
import os
import numpy as np
import torchaudio
import warnings
from pathlib import Path
from contextlib import redirect_stderr
import io
import json
import sys
import copy

from slicqt import model
from slicqt import torch_transforms

_mypath = Path(__file__).parent.resolve()
_pretrained_model = _mypath / "../pretrained-model"


def save_checkpoint(state: dict, is_best: bool, path: str, model_name: str):
    """Save checkpoint

    Args:
        state (dict): torch model state dict
        is_best (bool): if current model is about to be saved as best model
        path (str): model path
    """
    # save full checkpoint including optimizer
    torch.save(state, os.path.join(path, f"{model_name}.chkpnt"))
    if is_best:
        # save just the weights
        torch.save(state["state_dict"], os.path.join(path, f"{model_name}.pth"))


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class EarlyStopping(object):
    """Early Stopping Monitor"""

    def __init__(self, mode="min", min_delta=0, patience=10):
        self.mode = mode
        self.min_delta = min_delta
        self.patience = patience
        self.best = None
        self.num_bad_epochs = 0
        self.is_better = None
        self._init_is_better(mode, min_delta)

        if patience == 0:
            self.is_better = lambda a, b: True

    def step(self, metrics):
        if self.best is None:
            self.best = metrics
            return False

        if np.isnan(metrics):
            return True

        if self.is_better(metrics, self.best):
            self.num_bad_epochs = 0
            self.best = metrics
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs >= self.patience:
            return True

        return False

    def _init_is_better(self, mode, min_delta):
        if mode not in {"min", "max"}:
            raise ValueError("mode " + mode + " is unknown!")
        if mode == "min":
            self.is_better = lambda a, best: a < best - min_delta
        if mode == "max":
            self.is_better = lambda a, best: a > best + min_delta


def load_deoverlapnet(
    model_str_or_path=_pretrained_model,
    device="cpu",
    sample_rate=44100
):
    """Core model loader

    target model path can be either <target>.pth, or <target>-sha256.pth
    (as used on torchub)

    The loader either loads the models from a known model string
    as registered in the __init__.py or loads from custom configs.
    """
    model_path = Path(model_str_or_path).expanduser()

    # load model from disk
    with open(Path(model_path, "slicqt_deoverlapnet.json"), "r") as stream:
        results = json.load(stream)

    # need to configure an NSGT object to peek at its params to set up the neural network
    # e.g. M depends on the sllen which depends on fscale+fmin+fmax
    slicqt_base = torch_transforms.SliCQTBase(
        scale=results["args"]["fscale"],
        fbins=results["args"]["fbins"],
        fmin=results["args"]["fmin"],
        gamma=results["args"]["gamma"],
        fs=sample_rate,
        device=device
    )
    slicqt, islicqt = torch_transforms.make_filterbanks(slicqt_base)

    nb_channels = results["args"]["nb_channels"]
    seq_dur = results["args"]["seq_dur"]

    target_model_path = Path(model_path, "slicqt_deoverlapnet.pth")
    state = torch.load(target_model_path, map_location=device)

    jagged_slicq = slicqt_base.predict_input_size(1, nb_channels, seq_dur)
    jagged_slicq_mag, _ = torch_transforms.complex_2_magphase(jagged_slicq)

    deoverlapnet_model = model.DeOverlapNet(
        slicqt,
        jagged_slicq_mag,
    )

    deoverlapnet_model.load_state_dict(state, strict=False)
    deoverlapnet_model.to(device)

    return deoverlapnet_model, slicqt, islicqt
