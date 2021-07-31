from typing import Optional

import numpy as np
import torch
import torchaudio
from torch import Tensor
import torch.nn as nn
from .filtering import atan2
import warnings


from nsgt import NSGT, NSGT_sliced, BarkScale, MelScale, LogScale, VQLogScale

_SCALE = 'bark'
_BINS = 256
_FMIN = 22.6
_GAMMA = 25.
#_SEG_SAMPLES = 32768
_SEG_SAMPLES = 88200
SEG_DUR = float(_SEG_SAMPLES/44100)


def audio_segments(audio: Tensor):
    audio_segs = torch.split(audio, _SEG_SAMPLES, dim=-1)
    for audio_seg in audio_segs:
        audio_seg_len = audio_seg.shape[-1]
        if audio_seg_len < _SEG_SAMPLES:
            audio_seg = torch.nn.functional.pad(audio_seg, (0, _SEG_SAMPLES-audio_seg_len), mode='constant', value=0)

        yield audio_seg, audio_seg_len


def phasemix_sep(X, Ymag):
    Xphase = atan2(X[..., 1], X[..., 0])
    Ycomplex = torch.empty_like(X)

    Ycomplex[..., 0] = Ymag * torch.cos(Xphase)
    Ycomplex[..., 1] = Ymag * torch.sin(Xphase)
    return Ycomplex


def make_filterbanks(scale=_SCALE, fbins=_BINS, fmin=_FMIN, gamma=_GAMMA, sample_rate=44100.0, device="cuda"):
    if sample_rate != 44100.0:
        raise ValueError('i was lazy and harcoded a lot of 44100.0, forgive me')

    encoder = NSGT_SL(fs=sample_rate, device=device)
    decoder = INSGT_SL(fs=sample_rate, device=device)

    return encoder, decoder


class _NSGTBase(nn.Module):
    def __init__(self, scale=_SCALE, fbins=_BINS, fmin=_FMIN, gamma=_GAMMA, fs=44100, device="cuda"):
        super(_NSGTBase, self).__init__()

        self.fbins = fbins
        self.fbins_actual = self.fbins+1 # why 2 for mel with 113 bins?
        self.fmin = fmin
        self.fmax = fs/2

        self.scl = None
        if scale == 'bark':
            self.scl = BarkScale(self.fmin, self.fmax, self.fbins)
        elif scale == 'mel':
            self.scl = MelScale(self.fmin, self.fmax, self.fbins)
        elif scale == 'cqlog':
            self.scl = LogScale(self.fmin, self.fmax, self.fbins)
        elif scale == 'vqlog':
            self.scl = VQLogScale(self.fmin, self.fmax, self.fbins, gamma=gamma)
        else:
            raise ValueError(f'unsupported frequency scale {scale}')

        self.nsgt = NSGT(self.scl, fs, _SEG_SAMPLES, real=True, matrixform=True, multichannel=True, device=device)
        self.M = self.nsgt.ncoefs

    def max_bins(self, bandwidth): # convert hz bandwidth into bins
        if bandwidth is None:
            return None
        freqs, _ = self.scl()
        max_bin = min(np.argwhere(freqs > bandwidth))[0]
        return max_bin+1


class NSGT_SL(_NSGTBase):
    def __init__(self, *args, **kwargs):
        super(NSGT_SL, self).__init__(*args, **kwargs)

    def forward(self, x: Tensor) -> Tensor:
        """NSGT forward path
        Args:
            x (Tensor): audio waveform of
                shape (nb_samples, nb_channels, nb_timesteps)
        Returns:
            NSGT (Tensor): complex nsgt of
                shape (nb_samples, nb_channels, nb_bins_1, nb_bins_2, nb_frames, complex=2)
                last axis is stacked real and imaginary
        """
        shape = x.size()
        nb_samples, nb_channels, nb_timesteps = shape

        # pack batch
        x = x.view(-1, shape[-1])

        C = self.nsgt.forward(x)
        #I, F, T = C.shape

        nsgt_f = torch.view_as_real(C)

        # unpack batch
        nsgt_f = nsgt_f.view(shape[:-1] + nsgt_f.shape[-3:])

        return nsgt_f


class INSGT_SL(_NSGTBase):
    """NSGT backward path
    Args:
         NSGT (Tensor): complex stft of
             shape (nb_samples, nb_channels, nb_bins, nb_frames, complex=2)
             last axis is stacked real and imaginary
        OR
             shape (nb_samples, nb_targets, nb_channels, nb_bins, nb_frames, complex=2)
             last axis is stacked real and imaginary
    """
    def __init__(self, *args, **kwargs):
        super(INSGT_SL, self).__init__(*args, **kwargs)

    def forward(self, X: Tensor) -> Tensor:
        Xdims = len(X.shape)

        X = torch.view_as_complex(X)

        shape = X.shape

        if Xdims == 5:
            X = X.view(X.shape[0]*X.shape[1], *X.shape[2:])
        else:
            X = X.view(X.shape[0]*X.shape[1]*X.shape[2], *X.shape[3:])

        y = self.nsgt.backward(X)

        # unpack batch
        y = y.view(*shape[:-2], -1)

        return y


class ComplexNorm(nn.Module):
    r"""Compute the norm of complex tensor input.

    Extension of `torchaudio.functional.complex_norm` with mono

    Args:
        power (float): Power of the norm. (Default: `1.0`).
        mono (bool): Downmix to single channel after applying power norm
            to maximize
    """

    def __init__(self, power: float = 1.0, mono: bool = False):
        super(ComplexNorm, self).__init__()
        self.power = power
        self.mono = mono

    def forward(self, spec: Tensor) -> Tensor:
        """
        Args:
            spec: complex_tensor (Tensor): Tensor shape of
                `(..., complex=2)`

        Returns:
            Tensor: Power/Mag of input
                `(...,)`
        """
        # take the magnitude
        spec = torch.abs(torch.view_as_complex(spec))#, power=self.power)

        # downmix in the mag domain to preserve energy
        if self.mono:
            spec = torch.mean(spec, 1, keepdim=True)

        return spec
