from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter, ReLU, Tanh, BatchNorm1d, BatchNorm2d, Conv3d, Conv1d, ConvTranspose3d
import itertools
from .filtering import atan2
from .transforms import make_filterbanks, ComplexNorm, phasemix_sep, NSGTBase
from collections import defaultdict
import numpy as np

eps = 1.e-10


class OpenUnmix(nn.Module):
    """OpenUnmix Core spectrogram based separation module.

    Args:
        nb_bins (int): Number of input NSGT sliced tf bins (Default: `126`).
        nb_channels (int): Number of input audio channels (Default: `2`).
        nb_layers (int): Number of Bi-GRU layers (Default: `3`).
        unidirectional (bool): Use causal model useful for realtime purpose.
            (Default `False`)
        input_mean (ndarray or None): global data mean of shape `(nb_bins, )`.
            Defaults to zeros(nb_bins)
        input_scale (ndarray or None): global data mean of shape `(nb_bins, )`.
            Defaults to ones(nb_bins)
    """

    def __init__(
        self,
        nb_bins,
        M,
        max_bin=None,
        nb_channels=2,
        nb_layers=3,
        nb_conv_layers=3,
        unidirectional=False,
        input_mean=None,
        input_scale=None,
        debug=False
    ):
        super(OpenUnmix, self).__init__()
        self.debug = debug

        self.nb_output_bins = nb_bins
        self.nb_bins = max_bin if max_bin else nb_bins

        self.M = M

        # replace maxpooling with strided convolutions

        self.encoder = nn.ModuleList([
            Conv3d(nb_channels, 12, 3, stride=(1, 2, 3)),
            ReLU(),
            Conv3d(12, 20, 3, stride=(1, 1, 3)),
            ReLU(),
            Conv3d(20, 30, (1,3,3)),
            ReLU(),
            Conv3d(30, 40, (1,3,3)),
            ReLU()
        ])

        self.decoder = nn.ModuleList([
            ConvTranspose3d(40, 30, (1,3,3)),
            ReLU(),
            ConvTranspose3d(30, 20, (1,3,3)),
            ReLU(),
            ConvTranspose3d(20, 12, 3, stride=(1, 1, 3)),#, output_padding=(0, 0, 1)),
            ReLU(),
            ConvTranspose3d(12, nb_channels, (3, 3, 5), stride=(1, 2, 3), output_padding=(0, 0, 1)),
            ReLU(),
        ])

        if input_mean is not None:
            input_mean = (-input_mean).float()
        else:
            input_mean = torch.zeros(self.M*self.nb_output_bins)

        if input_scale is not None:
            input_scale = (1.0 / input_scale).float()
        else:
            input_scale = torch.ones(self.M*self.nb_output_bins)

        self.input_mean = Parameter(input_mean)
        self.input_scale = Parameter(input_scale)

    def freeze(self):
        # set all parameters as not requiring gradient, more RAM-efficient
        # at test time
        for p in self.parameters():
            p.requires_grad = False
        self.eval()

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: input spectrogram of shape
                `(nb_samples, nb_channels, nb_bins, nb_frames)`
        Returns:
            Tensor: filtered spectrogram of shape
                `(nb_samples, nb_channels, nb_bins, nb_frames)`
        """
        self.print(f'\n1. x.shape: {x.shape}')

        # crop frequency bins
        x = x[:, :, : self.nb_bins, :, :]

        self.print(f'2. x.shape: {x.shape}')

        # permute into samples, frames, channels, f_bins, m_bins
        x = x.permute(0, 3, 1, 2, 4)

        self.print(f'3. x.shape: {x.shape}')

        # combine slicq dims to apply scaling
        x_shape = x.shape
        x = x.reshape(*x_shape[:3], x_shape[-2]*x_shape[-1])

        # shift and scale input to mean=0.5 std=1 (across all bins)
        x = x + self.input_mean[: self.nb_bins*self.M]
        x = x * self.input_scale[: self.nb_bins*self.M]

        self.print(f'4. x.shape: {x.shape}')

        # separate slicq dims again to feed convnet
        x = x.reshape(*x_shape[:3], x_shape[-2], x_shape[-1])

        self.print(f'5. x.shape: {x.shape}')

        nb_samples, nb_frames, nb_channels, nb_f_bins, nb_m_bins = x.data.shape

        # reshape to samples, channels, frames, fbins, mbins for convolutional layers
        x = x.permute(0, 2, 1, 3, 4)

        self.print(f'6. x.shape: {x.shape}')

        for layer in self.encoder:
            self.print(f'\tenc. shape: {x.shape}')
            x = layer(x)

        self.print(f'7. x.shape: {x.shape}')

        for layer in self.decoder:
            self.print(f'\tdec. shape: {x.shape}')
            x = layer(x)

        self.print(f'8. x.shape: {x.shape}')

        # reshape to nb_samples, nb_frames, nb_channels, nb_f_bins, nb_m_bins = x.data.shape
        x = x.reshape(nb_samples, nb_channels, x.shape[-2], nb_frames, x.shape[-1])

        self.print(f'9. x.shape: {x.shape}')

        if self.debug:
            input()

        return x

    def print(self, msg):
        if self.debug:
            print(msg)
        return


class Separator(nn.Module):
    """
    Separator class to encapsulate all the stereo filtering
    as a torch Module, to enable end-to-end learning.

    Args:
        targets (dict of str: nn.Module): dictionary of target models
            the spectrogram models to be used by the Separator.
        niter (int): Number of EM steps for refining initial estimates in a
            post-processing stage. Zeroed if only one target is estimated.
            defaults to `1`.
        residual (bool): adds an additional residual target, obtained by
            subtracting the other estimated targets from the mixture,
            before any potential EM post-processing.
            Defaults to `False`.
        wiener_win_len (int or None): The size of the excerpts
            (number of frames) on which to apply filtering
            independently. This means assuming time varying stereo models and
            localization of sources.
            None means not batching but using the whole signal. It comes at the
            price of a much larger memory usage.
    """

    def __init__(
        self,
        target_models: dict,
        target_models_nsgt: dict,
        sample_rate: float = 44100.0,
        nb_channels: int = 2,
        device: str = "cpu",
    ):
        super(Separator, self).__init__()

        self.nsgts = defaultdict(dict)
        self.device = device

        # separate nsgt per model
        for name, nsgt_base in target_models_nsgt.items():
            nsgt, insgt = make_filterbanks(
                nsgt_base, sample_rate=sample_rate
            )

            self.nsgts[name]['nsgt'] = nsgt
            self.nsgts[name]['insgt'] = insgt

        self.complexnorm = ComplexNorm(mono=nb_channels == 1)

        # registering the targets models
        self.target_models = nn.ModuleDict(target_models)
        # adding till https://github.com/pytorch/pytorch/issues/38963
        self.nb_targets = len(self.target_models)
        # get the sample_rate as the sample_rate of the first model
        # (tacitly assume it's the same for all targets)
        self.register_buffer("sample_rate", torch.as_tensor(sample_rate))

    def freeze(self):
        # set all parameters as not requiring gradient, more RAM-efficient
        # at test time
        for p in self.parameters():
            p.requires_grad = False
        self.eval()

    def forward(self, audio: Tensor) -> Tensor:
        """Performing the separation on audio input

        Args:
            audio (Tensor): [shape=(nb_samples, nb_channels, nb_timesteps)]
                mixture audio waveform

        Returns:
            Tensor: stacked tensor of separated waveforms
                shape `(nb_samples, nb_targets, nb_channels, nb_timesteps)`
        """

        nb_sources = self.nb_targets
        nb_samples = audio.shape[0]

        # getting the STFT of mix:
        # (nb_samples, nb_channels, nb_bins, nb_frames, 2)

        # initializing spectrograms variable
        estimates = torch.zeros(audio.shape + (nb_sources,), dtype=audio.dtype, device=self.device)

        for j, (target_name, target_module) in enumerate(self.target_models.items()):
            print(f'separating {target_name}')

            nsgt = self.nsgts[target_name]['nsgt']
            insgt = self.nsgts[target_name]['insgt']

            X = nsgt(audio)

            Xmag = self.complexnorm(X)

            # apply current model to get the source magnitude spectrogram

            if target_name == 'bass':
                # apply it in halves to avoid oom for the larger bass model
                Xmag_split = torch.split(Xmag, int(np.ceil(Xmag.shape[-2]/2)), dim=-2)

                Xmag_0 = Xmag_split[0]
                Xmag_1 = Xmag_split[1]

                Ymag_0 = target_module(Xmag_0.detach().clone())
                Ymag_1 = target_module(Xmag_1.detach().clone())

                Ymag = torch.cat([Ymag_0, Ymag_1], dim=-2)
            else:
                Ymag = target_module(Xmag.detach().clone())

            Y = phasemix_sep(X, Ymag)
            y = insgt(Y, audio.shape[-1])

            estimates[..., j] = y

        # getting to (nb_samples, nb_targets, nb_channels, nb_samples)
        estimates = estimates.permute(0, 3, 1, 2).contiguous()

        return estimates

    def to_dict(self, estimates: Tensor, aggregate_dict: Optional[dict] = None) -> dict:
        """Convert estimates as stacked tensor to dictionary

        Args:
            estimates (Tensor): separated targets of shape
                (nb_samples, nb_targets, nb_channels, nb_timesteps)
            aggregate_dict (dict or None)

        Returns:
            (dict of str: Tensor):
        """
        estimates_dict = {}
        for k, target in enumerate(self.target_models):
            estimates_dict[target] = estimates[:, k, ...]

        if aggregate_dict is not None:
            new_estimates = {}
            for key in aggregate_dict:
                new_estimates[key] = torch.tensor(0.0)
                for target in aggregate_dict[key]:
                    new_estimates[key] = new_estimates[key] + estimates_dict[target]
            estimates_dict = new_estimates
        return estimates_dict
