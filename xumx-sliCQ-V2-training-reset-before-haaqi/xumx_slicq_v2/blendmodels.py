from typing import Optional, Union, Tuple
import sys
from tqdm import trange
from pathlib import Path
import torch
import json
import copy
from torch import Tensor
import torch.nn as nn
from .models import Unmix, _SlicedUnmix
import norbert
from .transforms import (
    TorchSTFT,
    TorchISTFT,
    ComplexNorm,
    NSGTBase,
    make_filterbanks,
    phasemix_sep,
)
from .separator import load_target_models



if __name__ == '__main__':
    device = torch.device("cpu")

    print("Blending MSE + MSE-SDR models...")

    mse_model_path = "/xumx-sliCQ-V2/pretrained_model/mse"
    mse_model_path = Path(mse_model_path)

    # when path exists, we assume its a custom model saved locally
    assert mse_model_path.exists()

    mse_xumx_model, _ = load_target_models(
        mse_model_path,
        sample_rate=44100.,
        device=device,
    )

    msesdr_model_path = "/xumx-sliCQ-V2/pretrained_model/mse-sdr"
    msesdr_model_path = Path(msesdr_model_path)

    # when path exists, we assume its a custom model saved locally
    assert msesdr_model_path.exists()

    msesdr_xumx_model, _ = load_target_models(
        msesdr_model_path,
        sample_rate=44100.,
        device=device,
    )

    # copy bass cdae from mse-sdr model into mse model
    print("Deep-copying single cdae from mse-sdr model into mse model")

    for i in range(len(mse_xumx_model.sliced_umx)):
        sliced_obj = mse_xumx_model.sliced_umx[i]

        if type(sliced_obj) == _SlicedUnmix:
            # second-last cdae i.e. index 2 is the 'other' cdae
            mse_xumx_model.sliced_umx[i].cdaes[2] = copy.deepcopy(msesdr_xumx_model.sliced_umx[i].cdaes[2])

    blend_model_path = "/xumx-sliCQ-V2/pretrained_model/blend"
    blend_model_path = Path(blend_model_path)

    print("Now saving blended model")
    torch.save(mse_xumx_model.state_dict(), Path(blend_model_path / "xumx_slicq_v2.pth"))
