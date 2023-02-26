from typing import Optional, Union, Tuple
import sys
from tqdm import trange
from pathlib import Path
import torch
import json
import copy
from torch import Tensor
import torch.nn as nn
from .model import Unmix, _SlicedUnmix
import norbert
from .transforms import (
    ComplexNorm,
    NSGTBase,
    make_filterbanks,
)
from .separator import load_target_models



if __name__ == '__main__':
    device = torch.device("cpu")

    print("Blending bottleneck + no-bottleneck models...")

    model_path_1 = "/xumx-sliCQ-V2/pretrained_model/"
    model_path_1 = Path(model_path_1)

    # when path exists, we assume its a custom model saved locally
    assert model_path_1.exists()

    xumx_model_1, _ = load_target_models(
        model_path_1,
        sample_rate=44100.,
        device=device,
    )

    model_path_2 = "/xumx-sliCQ-V2/pretrained_model_bottleneck/"
    model_path_2 = Path(model_path_2)

    # when path exists, we assume its a custom model saved locally
    assert model_path_2.exists()

    xumx_model_2, _ = load_target_models(
        model_path_2,
        sample_rate=44100.,
        device=device,
    )

    # copy vocals, bass, other cdae from no-bottleneck model into bottleneck model
    print("Deep-copying cdaes from model 1 to model 2")

    for i in range(len(xumx_model_1.sliced_umx)):
        xumx_model_2.sliced_umx[i].cdaes[0] = copy.deepcopy(xumx_model_1.sliced_umx[i].cdaes[0])
        xumx_model_2.sliced_umx[i].cdaes[1] = copy.deepcopy(xumx_model_1.sliced_umx[i].cdaes[1])
        xumx_model_2.sliced_umx[i].cdaes[2] = copy.deepcopy(xumx_model_1.sliced_umx[i].cdaes[2])

    blend_model_path = "/xumx-sliCQ-V2/pretrained_model_blended/"
    blend_model_path = Path(blend_model_path)

    print("Now saving blended model")
    torch.save(xumx_model_2.state_dict(), Path(blend_model_path / "xumx_slicq_v2.pth"))
