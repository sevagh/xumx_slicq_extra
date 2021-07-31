from typing import Optional

import numpy as np
import torch
import torchaudio
from torch import Tensor
import torch.nn as nn

from nsgt import NSGT_sliced, BarkScale


def make_filterbanks(sample_rate=44100.0, device="cuda"):
    n = NSGT(fs=sample_rate, device=device)

    encoder = NSGT_SL(n)
    decoder = INSGT_SL(n)

    return encoder, decoder


class NSGT:
    def __init__(self, fs=44100, fmin=78.0, fbins=125, sllen=9216, trlen=2304, device="cuda"):
        self.fbins = fbins
        self.sllen = sllen

        scl = BarkScale(fmin, fs/2, fbins)

        self.nsgt = NSGT_sliced(scl, sllen, trlen, fs, real=True, matrixform=True, multichannel=True, device=device)

        # the 126*304 comes from here
        self.fdim1 = fbins+1
        self.fdim2 = int(self.nsgt.coef_factor*sllen)


class NSGT_SL(nn.Module):
    def __init__(self, nsgt):
        super(NSGT_SL, self).__init__()
        self.nsgt = nsgt

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
        T, I, F1, F2 = C.shape

        # first, moveaxis T, I, F1, F2 to I, F1, F2, T
        C = torch.moveaxis(C, 0, -1)

        # then, combine it to final shape I x F x T
        C = C.reshape(I, F1*F2, T)
        # will have to uncombine it in the inversion

        nsgt_f = torch.view_as_real(C)

        # unpack batch
        nsgt_f = nsgt_f.view(shape[:-1] + nsgt_f.shape[-3:])

        return nsgt_f


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


    def forward(self, X: Tensor, length: int) -> Tensor:
        if len(X.shape) != 6:
            raise ValueError('only tensor with all targets, not {0}'.format(len(shape)))

        X = torch.view_as_complex(X)

        shape = X.shape

        # pack batch
        X = X.view(-1, *shape[-2:])

        # nsgt - rearrange [packed-channels] x F x T to  [packed-channels] x F1 x F2 x T
        C  = X.reshape(X.shape[0], self.nsgt.fdim1, self.nsgt.fdim2, shape[-1])

        # finally moveaxis back into into T x [packed-channels] x F1 x F2
        C = torch.moveaxis(C, -1, 0)

        y = self.nsgt.nsgt.backward(C, length)

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
        spec = torchaudio.functional.complex_norm(spec, power=self.power)

        # downmix in the mag domain to preserve energy
        if self.mono:
            spec = torch.mean(spec, 1, keepdim=True)

        return spec
