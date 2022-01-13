import pytest
import numpy as np
import torch
from xumx_slicq_22 import transforms
from xumx_slicq_22 import utils
import auraloss

import matplotlib.pyplot as plt


# try some durations
@pytest.fixture(params=[4096, 44100, int(44100*2)])
def nb_timesteps(request):
    return int(request.param)


@pytest.fixture(params=[1, 2])
def nb_channels(request):
    return request.param


@pytest.fixture(params=[1, 5, 13, 56])
def nb_samples(request):
    return request.param

@pytest.fixture
def audio(request, nb_samples, nb_channels, nb_timesteps):
    return torch.rand((nb_samples, nb_channels, nb_timesteps), dtype=torch.float64, device="cpu")


def test_nsgt_cpu_fwd_inv(audio):
    nsgt, insgt = transforms.make_filterbanks_slicqt(transforms.NSGTBase(scale='mel', fbins=200, fmin=32.9, device="cpu"))

    X = nsgt(audio)

    out = insgt(X, length=audio.shape[-1])

    print(auraloss.time.SNRLoss()(audio, out))
    assert np.sqrt(np.mean((audio.detach().numpy() - out.detach().numpy()) ** 2)) < 1e-6


def test_nsgt_cpu_fwd_inv_ola_interpolate_deoverlapnet_harmonic(audio):
    nsgt, insgt = transforms.make_filterbanks_slicqt(
        transforms.NSGTBase(scale='bark', fbins=287, fmin=29.8, device="cpu")
    )

    cnorm = transforms.ComplexNormSliCQT().to(device="cpu")
 
    X = nsgt(audio)
    Xmag = cnorm(X)
    shapes = [X_.shape for X_ in Xmag]
    nb_slices = Xmag[0].shape[-2]

    deoverlapnet = utils.load_deoverlapnet(device="cpu")

    Xmag_interp = nsgt.interpolate(Xmag)
    Xmag_ola = nsgt.overlap_add(Xmag_interp)
    Xmag_deoverlapnet, _ = deoverlapnet(harmonic_inputs=(Xmag_ola, nb_slices, shapes))

    Xrecon_complex = transforms.phasemix_sep(X, Xmag_deoverlapnet)
    out = insgt(Xrecon_complex, length=audio.shape[-1])

    print(auraloss.time.SNRLoss()(audio, out))
    assert np.sqrt(np.mean((audio.detach().numpy() - out.detach().numpy()) ** 2)) < 1e-6


def test_nsgt_cpu_fwd_inv_ola_interpolate_deoverlapnet_percussive(audio):
    nsgt, insgt = transforms.make_filterbanks_slicqt(
        transforms.NSGTBase(scale='bark', fbins=265, fmin=33.4, device="cpu")
    )

    cnorm = transforms.ComplexNormSliCQT().to(device="cpu")
 
    X = nsgt(audio)
    Xmag = cnorm(X)
    shapes = [X_.shape for X_ in Xmag]
    nb_slices = Xmag[0].shape[-2]

    deoverlapnet = utils.load_deoverlapnet(device="cpu")

    Xmag_interp = nsgt.interpolate(Xmag)
    Xmag_ola = nsgt.overlap_add(Xmag_interp)
    _, Xmag_deoverlapnet = deoverlapnet(percussive_inputs=(Xmag_ola, nb_slices, shapes))

    Xrecon_complex = transforms.phasemix_sep(X, Xmag_deoverlapnet)
    out = insgt(Xrecon_complex, length=audio.shape[-1])

    print(auraloss.time.SNRLoss()(audio, out))
    assert np.sqrt(np.mean((audio.detach().numpy() - out.detach().numpy()) ** 2)) < 1e-6


def test_nsgt_cuda_fwd_inv(audio):
    audio = audio.to(device="cuda")

    nsgt, insgt = transforms.make_filterbanks_slicqt(transforms.NSGTBase(scale='mel', fbins=200, fmin=32.9, device="cuda"))

    X = nsgt(audio)

    out = insgt(X, length=audio.shape[-1])

    print(auraloss.time.SNRLoss()(audio, out))
    assert np.sqrt(np.mean((audio.detach().cpu().numpy() - out.detach().cpu().numpy()) ** 2)) < 1e-6


def test_nsgt_cuda_fwd_inv_ola_interpolate_deoverlapnet_harmonic(audio):
    audio = audio.to(device="cuda")

    nsgt, insgt = transforms.make_filterbanks_slicqt(
        transforms.NSGTBase(scale='bark', fbins=287, fmin=29.8, device="cuda")
    )

    cnorm = transforms.ComplexNormSliCQT().to(device="cuda")
 
    X = nsgt(audio)
    Xmag = cnorm(X)
    shapes = [X_.shape for X_ in Xmag]
    nb_slices = Xmag[0].shape[-2]

    deoverlapnet = utils.load_deoverlapnet(device="cuda")

    Xmag_interp = nsgt.interpolate(Xmag)
    Xmag_ola = nsgt.overlap_add(Xmag_interp)
    Xmag_deoverlapnet, _ = deoverlapnet(harmonic_inputs=(Xmag_ola, nb_slices, shapes))

    Xrecon_complex = transforms.phasemix_sep(X, Xmag_deoverlapnet)
    out = insgt(Xrecon_complex, length=audio.shape[-1])

    print(auraloss.time.SNRLoss()(audio, out))
    assert np.sqrt(np.mean((audio.detach().cpu().numpy() - out.detach().cpu().numpy()) ** 2)) < 1e-6


def test_nsgt_cuda_fwd_inv_ola_interpolate_deoverlapnet_percussive(audio):
    audio = audio.to(device="cuda")

    nsgt, insgt = transforms.make_filterbanks_slicqt(
        transforms.NSGTBase(scale='bark', fbins=265, fmin=33.4, device="cuda")
    )

    cnorm = transforms.ComplexNormSliCQT().to(device="cuda")
 
    X = nsgt(audio)
    Xmag = cnorm(X)
    shapes = [X_.shape for X_ in Xmag]
    nb_slices = Xmag[0].shape[-2]

    deoverlapnet = utils.load_deoverlapnet(device="cuda")

    Xmag_interp = nsgt.interpolate(Xmag)
    Xmag_ola = nsgt.overlap_add(Xmag_interp)
    _, Xmag_deoverlapnet = deoverlapnet(percussive_inputs=(Xmag_ola, nb_slices, shapes))

    Xrecon_complex = transforms.phasemix_sep(X, Xmag_deoverlapnet)
    out = insgt(Xrecon_complex, length=audio.shape[-1])

    print(auraloss.time.SNRLoss()(audio, out))
    assert np.sqrt(np.mean((audio.detach().cpu().numpy() - out.detach().cpu().numpy()) ** 2)) < 1e-6


import pytest
pytest.main(["-s", "tests/test_transforms.py::test_nsgt_cpu_fwd_inv"])
