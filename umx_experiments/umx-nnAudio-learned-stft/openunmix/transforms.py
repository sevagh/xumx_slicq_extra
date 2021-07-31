from typing import Optional

import torch
import nnAudio
import nnAudio.Spectrogram

import torchaudio
from torch import Tensor
import torch.nn as nn
from .filtering import atan2

try:
    from asteroid_filterbanks.enc_dec import Encoder, Decoder
    from asteroid_filterbanks.transforms import to_torchaudio, from_torchaudio
    from asteroid_filterbanks import torch_stft_fb
except ImportError:
    pass


def phasemix_sep(X, Ymag):
    Xphase = atan2(X[..., 1], X[..., 0])
    Ycomplex = torch.empty_like(X)
    Ycomplex[..., 0] = Ymag * torch.cos(Xphase)
    Ycomplex[..., 1] = Ymag * torch.sin(Xphase)
    return Ycomplex


def make_filterbanks(n_fft=4096, n_hop=1024, center=False, sample_rate=44100.0, fmin=0, fmax=16000):
    encoder = nnAudio.Spectrogram.STFT(n_fft=n_fft, hop_length=n_hop, sr=sample_rate, freq_scale='linear', fmin=fmin, fmax=fmax, center=center, trainable=True, iSTFT=True)

    return encoder


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
        # clone due to some gradient in-place bug...
        spec2 = torch.abs(torch.view_as_complex(spec)).detach().clone()
        spec = spec2
        spec = torch.pow(spec, self.power)

        spec = torch.unsqueeze(spec, dim=1)

        return spec
