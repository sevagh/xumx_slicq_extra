import pytest
import numpy as np
import torch
from openunmix import transforms

import matplotlib.pyplot as plt


# try some durations
@pytest.fixture(params=[4096, 44100, int(44100*2)])
def nb_timesteps(request):
    return int(request.param)


@pytest.fixture(params=[1, 2, 3, 4, 9])
def nb_channels(request):
    return request.param


@pytest.fixture(params=[1, 2, 3, 5, 7])
def nb_samples(request):
    return request.param

@pytest.fixture
def audio(request, nb_samples, nb_channels, nb_timesteps):
    return torch.rand((nb_samples, nb_channels, nb_timesteps))


def test_nsgt_fwd_inv_cpu(audio):
    audio = audio.detach().cpu()

    nsgt, insgt = transforms.make_filterbanks(device="cpu")

    audio_seg_gen = transforms.audio_segments(audio)
    out = []

    for audio_seg, crop_len in audio_seg_gen:
        X = nsgt(audio_seg)
        shape = X.shape
        # add fake target of size 1
        X = X.reshape(shape[0], 1, *shape[1:])
        out.append(insgt(X)[..., : crop_len])

    out = torch.cat([out_ for out_ in out], dim=-1)

    # remove fake target of size 1
    out = torch.squeeze(out, dim=1)

    assert np.sqrt(np.mean((audio.detach().numpy() - out.detach().numpy()) ** 2)) < 1e-6


def test_nsgt_fwd_inv_cuda(audio):
    audio = audio.detach().clone().to(torch.device("cuda"))

    nsgt, insgt = transforms.make_filterbanks(device="cuda")

    audio_seg_gen = transforms.audio_segments(audio)
    out = []

    for audio_seg, crop_len in audio_seg_gen:
        X = nsgt(audio_seg)
        shape = X.shape
        # add fake target of size 1
        X = X.reshape(shape[0], 1, *shape[1:])
        out.append(insgt(X)[..., : crop_len])

    recon = torch.cat([out_ for out_ in out], dim=-1)
    # remove fake target of size 1
    recon = torch.squeeze(recon, dim=1)

    assert torch.sqrt(torch.mean((audio.detach() - recon.detach()) ** 2)) < 1e-6


def test_nsgt_fwd_inv_cuda_no_fake_targets(audio):
    audio = audio.detach().clone().to(torch.device("cuda"))

    nsgt, insgt = transforms.make_filterbanks(device="cuda")

    audio_seg_gen = transforms.audio_segments(audio)
    out = []

    for audio_seg, crop_len in audio_seg_gen:
        X = nsgt(audio_seg)
        shape = X.shape
        out.append(insgt(X)[..., : crop_len])

    out = torch.cat([out_ for out_ in out], dim=-1)

    assert torch.sqrt(torch.mean((audio.detach() - out.detach()) ** 2)) < 1e-6
