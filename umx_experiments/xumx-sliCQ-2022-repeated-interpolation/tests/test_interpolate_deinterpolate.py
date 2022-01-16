import pytest
import numpy as np
import torch
from xumx_slicq import transforms

import matplotlib.pyplot as plt


# try some durations
@pytest.fixture(params=[4096, 44100, int(44100*3.5)])
def nb_timesteps(request):
    return int(request.param)


@pytest.fixture(params=[1, 2, 3, 4, 9])
def nb_channels(request):
    return request.param


@pytest.fixture(params=[1, 2, 3, 5, 7, 13, 56])
def nb_samples(request):
    return request.param

@pytest.fixture
def audio(request, nb_samples, nb_channels, nb_timesteps):
    return torch.rand((nb_samples, nb_channels, nb_timesteps))


def test_nsgt_interp_deinterp_cpu_slicq1(audio):
    audio = audio.detach().cpu()

    nsgt, _ = transforms.make_filterbanks(transforms.NSGTBase('cqlog', 80, 20, device="cpu"))

    X = nsgt(audio)
    cnorm = transforms.ComplexNorm()

    Xmag = cnorm(X)

    X_ola = transforms.overlap_add_slicq(Xmag)
    ragged_shapes = [X_ola_.shape for X_ola_ in X_ola]
    X_interp = transforms.repeated_interpolation(X_ola)

    X_deinterp = transforms.repeated_deinterpolation(X_interp, ragged_shapes)

    # should be equivalent to the overlap-added slicqt after an interp->deinterp round trip
    err = 0.
    for i, X_ola_block in enumerate(X_ola):
        err += np.sqrt(np.mean((X_ola_block.detach().numpy() - X_deinterp[i].detach().numpy()) ** 2)) 
    err /= len(X_ola)
    print(f'interp->deinterp error: {err}')

    assert err < 1e-6


def test_nsgt_interp_deinterp_cpu_slicq2(audio):
    audio = audio.detach().cpu()

    nsgt, _ = transforms.make_filterbanks(transforms.NSGTBase('bark', 262, 32.9, device="cpu"))

    X = nsgt(audio)
    cnorm = transforms.ComplexNorm()

    Xmag = cnorm(X)

    X_ola = transforms.overlap_add_slicq(Xmag)
    ragged_shapes = [X_ola_.shape for X_ola_ in X_ola]
    X_interp = transforms.repeated_interpolation(X_ola)

    X_deinterp = transforms.repeated_deinterpolation(X_interp, ragged_shapes)

    # should be equivalent to the overlap-added slicqt after an interp->deinterp round trip
    err = 0.
    for i, X_ola_block in enumerate(X_ola):
        err += np.sqrt(np.mean((X_ola_block.detach().numpy() - X_deinterp[i].detach().numpy()) ** 2)) 
    err /= len(X_ola)
    print(f'interp->deinterp error: {err}')

    assert err < 1e-6


import pytest
pytest.main(["-s", "tests/test_interpolate_deinterpolate.py::test_nsgt_interp_deinterp_cpu_slicq1"])
