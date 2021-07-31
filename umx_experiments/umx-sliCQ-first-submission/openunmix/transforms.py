from typing import Optional

import numpy as np
import torch
import torchaudio
from torch import Tensor
import torch.nn as nn
from .filtering import atan2

from .nsgt import NSGT_sliced, BarkScale, MelScale, LogScale, VQLogScale, OctScale


def phasemix_sep(X, Ymag):
    Xphase = atan2(X[..., 1], X[..., 0])
    Ycomplex = torch.empty_like(X)

    Ycomplex[..., 0] = Ymag * torch.cos(Xphase)
    Ycomplex[..., 1] = Ymag * torch.sin(Xphase)
    return Ycomplex


def make_filterbanks(nsgt_base, sample_rate=44100.0):
    if sample_rate != 44100.0:
        raise ValueError('i was lazy and harcoded a lot of 44100.0, forgive me')

    encoder = NSGT_SL(nsgt_base)
    decoder = INSGT_SL(nsgt_base)

    return encoder, decoder


class NSGTBase(nn.Module):
    def __init__(self, scale, fbins, fmin, sllen=None, fs=44100, device="cuda", gamma=25.):
        super(NSGTBase, self).__init__()
        self.fbins = fbins
        self.fmin = fmin
        self.fmax = fs/2
        self.gamma = gamma

        self.scl = None
        if scale == 'bark':
            self.scl = BarkScale(self.fmin, self.fmax, self.fbins)
        elif scale == 'mel':
            self.scl = MelScale(self.fmin, self.fmax, self.fbins)
        elif scale == 'cqlog':
            self.scl = LogScale(self.fmin, self.fmax, self.fbins)
        elif scale == 'vqlog':
            self.scl = VQLogScale(self.fmin, self.fmax, self.fbins, self.gamma)
        elif scale == 'oct':
            self.scl = OctScale(self.fmin, self.fmax, self.fbins)
        else:
            raise ValueError(f'unsupported frequency scale {scale}')

        min_sllen = self.scl.suggested_sllen(fs)

        if sllen is not None:
            if sllen < min_sllen:
                print('damn')
                raise ValueError(f"slice length is too short for desired frequency scale, need {min_sllen}")

            self.sllen = sllen
        else:
            self.sllen = min_sllen

        trlen = self.sllen//4
        trlen = trlen + -trlen % 2 # make trlen divisible by 2
        self.trlen = trlen

        self.nsgt = NSGT_sliced(self.scl, self.sllen, self.trlen, fs, real=True, matrixform=False, multichannel=True, device=device)

        maxidx = np.argmax(self.nsgt.M)
        lh = self.nsgt.M[:maxidx+1]
        rh = self.nsgt.M[maxidx:]

        direction = 'increasing'
        changes = 0
        prev_m = self.nsgt.M[0]
        for i, m in enumerate(self.nsgt.M[1:]):
            if m < prev_m and direction != 'decreasing':
                direction = 'decreasing'
                changes += 1
            elif m > prev_m and direction != 'increasing':
                direction = 'increasing'
                changes += 1

            prev_m = m

        if changes > 1:
            raise ValueError(f"time resolution is not monotonically increasing then decreasing")

        self.fs = fs
        self.fbins_actual = self.nsgt.fbins_actual

    def max_bins(self, bandwidth): # convert hz bandwidth into bins
        if bandwidth is None:
            return None
        freqs, _ = self.scl()
        max_bin = min(np.argwhere(freqs > bandwidth))[0]
        return max_bin+1

    def predict_input_size(self, batch_size, nb_channels, seq_dur_s):
        fwd = NSGT_SL(self)

        x = torch.rand((batch_size, nb_channels, int(seq_dur_s*self.fs)), dtype=torch.float32)
        shape = x.size()
        nb_samples, nb_channels, nb_timesteps = shape

        nsgt_f = fwd(x)
        return nsgt_f

    def _apply(self, fn):
        self.nsgt._apply(fn)
        return self


class NSGT_SL(nn.Module):
    def __init__(self, nsgt):
        super(NSGT_SL, self).__init__()
        self.nsgt = nsgt

    def _apply(self, fn):
        self.nsgt._apply(fn)
        return self

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

        C = self.nsgt.nsgt.forward((x,))

        for time_bucket, nsgt_f in C.items():
            nsgt_f = torch.moveaxis(nsgt_f, 0, -2)
            nsgt_f = torch.view_as_real(nsgt_f)
            # unpack batch
            nsgt_f = nsgt_f.view(shape[:-1] + nsgt_f.shape[-4:])
            C[time_bucket] = nsgt_f

        return C


class INSGT_SL(nn.Module):
    '''
    wrapper for torch.istft to support batches
    Args:
         NSGT (Tensor): complex stft of
             shape (nb_samples, nb_channels, nb_bins, nb_frames, complex=2)
             last axis is stacked real and imaginary
        OR
             shape (nb_samples, nb_targets, nb_channels, nb_bins, nb_frames, complex=2)
             last axis is stacked real and imaginary
     '''
    def __init__(self, nsgt):
        super(INSGT_SL, self).__init__()
        self.nsgt = nsgt

    def _apply(self, fn):
        self.nsgt._apply(fn)
        return self

    def forward(self, X_dict, length: int) -> Tensor:
        X_complex = {}
        for time_bucket, X in X_dict.items():
            Xshape = len(X.shape)

            X = torch.view_as_complex(X)

            shape = X.shape

            if Xshape == 6:
                X = X.view(X.shape[0]*X.shape[1], *X.shape[2:])
            else:
                X = X.view(X.shape[0]*X.shape[1]*X.shape[2], *X.shape[3:])

            # moveaxis back into into T x [packed-channels] x F1 x F2
            X = torch.moveaxis(X, -2, 0)

            X_complex[time_bucket] = X

        y = self.nsgt.nsgt.backward(X_complex, length)

        # unpack batch
        y = y.view(*shape[:-3], -1)

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

    def forward(self, spec):
        # take the magnitude

        ret = {}
        for time_bucket, C_block in spec.items():
            C_block = torch.pow(torch.abs(torch.view_as_complex(C_block)), self.power)

            # downmix in the mag domain to preserve energy
            if self.mono:
                C_block = torch.mean(C_block, 1, keepdim=True)
            ret[time_bucket] = C_block

        return ret
