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
    def __init__(self, fs=44100, fmin=78.0, fmax=None, fbins=125, sllen=9216, trlen=2304, device="cuda"):
        self.device = device
        self.fbins = fbins
        self.sllen = sllen

        if not fmax:
            fmax = fs/2

        scl = BarkScale(fmin, fmax, fbins)

        self.nsgt = NSGT_sliced(scl, sllen, trlen, fs, real=True, matrixform=False, multichannel=True, reducedform=True, device=device)

        self.fdim = fbins+1

        # M time steps per frequency bin
        self.tdim = int(self.nsgt.coef_factor*sllen)


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
        print('x: {0}'.format(x.shape))
        shape = x.size()
        nb_samples, nb_channels, nb_timesteps = shape

        # pack batch
        x = x.view(-1, shape[-1])

        print('x: {0}'.format(x.shape))

        C = self.nsgt.nsgt.forward((x,))
        print("post-nsgt X.shape: {0}".format(C.shape))
        T, I, F1, F2 = C.shape

        # first, moveaxis T, I, F1, F2 to I, F1, F1, T
        C = torch.moveaxis(C, 0, -1)
        print('C shape: {0}'.format(C.shape))

        nsgt_f = torch.view_as_real(C)

        # unpack batch
        nsgt_f = nsgt_f.view(shape[:-1] + nsgt_f.shape[-4:])

        print('returning: {0}'.format(nsgt_f.shape))

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
        print('insgt: {0}'.format(X.shape))

        pack_batch_dim = 3
        if len(X.shape) == 6:
            pack_batch_dim = 2
            #raise ValueError('only tensor with all targets, not {0}'.format(len(X.shape)))

        X = torch.view_as_complex(X)

        shape = X.shape

        # pack batch
        X = X.view(-1, *shape[-pack_batch_dim:])

        print('insgt: {0}'.format(X.shape))

        # moveaxis back into into T x [packed-channels] x F1 x F2
        X = torch.moveaxis(X, -1, 0)

        print("pre-insgt X.shape: {0}".format(X.shape))

        y = self.nsgt.nsgt.backward(X, length)

        # unpack batch
        y = y.view(*shape[:-pack_batch_dim], -1)

        print('return: {0}'.format(y.shape))

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
