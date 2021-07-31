from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter, ReLU, Tanh, BatchNorm1d, GRU, Linear, Conv2d, ConvTranspose2d, Dropout, BatchNorm2d
import itertools
from .filtering import atan2
from .transforms import make_filterbanks, ComplexNorm, phasemix_sep, audio_segments
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
        nb_channels=2,
        nb_layers=3,
        unidirectional=False,
        input_mean=None,
        input_scale=None,
        debug=False
    ):
        super(OpenUnmix, self).__init__()
        self.debug = debug
        hidden_size = 2*nb_bins
        self.nb_bins = nb_bins

        self.encoder = nn.ModuleList([
            Conv2d(nb_channels, 12, 3, stride=(5, 3)),
            ReLU(),
            BatchNorm2d(12),
            Conv2d(12, 20, 3, stride=(5, 1)),
            ReLU(),
            BatchNorm2d(20),
            Conv2d(20, 30, 3),
            ReLU(),
            BatchNorm2d(30),
            Conv2d(30, 40, 3),
            ReLU(),
            BatchNorm2d(40),
        ])

        self.decoder = nn.ModuleList([
            ConvTranspose2d(40, 30, 3),
            ReLU(),
            BatchNorm2d(30),
            ConvTranspose2d(30, 20, 3),
            ReLU(),
            BatchNorm2d(20),
            ConvTranspose2d(20, 12, 3, stride=(5, 1), output_padding=(4, 0)),
            ReLU(),
            BatchNorm2d(12),
            ConvTranspose2d(12, nb_channels, 3, stride=(5, 3), output_padding=(4, 2)),
            #ReLU(),
        ])

        if input_mean is not None:
            input_mean = (-input_mean).float()
        else:
            input_mean = torch.zeros(self.nb_bins)

        if input_scale is not None:
            input_scale = (1.0 / input_scale).float()
        else:
            input_scale = torch.ones(self.nb_bins)

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
        mix = x.detach().clone()

        self.print(f'\n1. x.shape: {x.shape}')
        self.print(f'1. input_mean.shape: {self.input_mean.shape}')
        self.print(f'1. input_scale.shape: {self.input_scale.shape}')

        # permute to put time before frequency
        x = x.permute(0, 1, 3, 2)

        # get nsgt spectrogram shape
        nb_samples, nb_channels, nb_frames, nb_bins = x.data.shape

        # shift and scale input to mean=0 std=1 (across all bins)
        x = x + self.input_mean
        x = x * self.input_scale

        self.print(f'2. x.shape: {x.shape}')

        for layer in self.encoder:
            self.print(f'\tenc. x.shape: {x.shape}')
            x = layer(x)
            self.print(f'\tenc. x.shape: {x.shape}')

        self.print(f'3. x.shape: {x.shape}')

        for layer in self.decoder:
            self.print(f'\tdec. x.shape: {x.shape}')
            x = layer(x)
            self.print(f'\tdec. x.shape: {x.shape}')

        self.print(f'4. x.shape: {x.shape}')

        # permute back to original dims
        x = x.permute(0, 1, 3, 2)

        self.print(f'5. x.shape: {x.shape}')
        self.print(f'6. mix.shape: {mix.shape}')

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
        sample_rate: float = 44100.0,
        nb_channels: int = 2,
        device: str = "cpu",
    ):
        super(Separator, self).__init__()

        self.device = device

        # seq dur - make this customizeable
        # batch the seq dur make inference faster - multiple 6-second
        self.nsgt, self.insgt = make_filterbanks(
            sample_rate=sample_rate, device=device
        )

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

        audio_seg_gen = audio_segments(audio)

        estimates = []
        for _ in range(self.nb_targets):
            estimates.append([])

        total = 0
        for i, (audio_seg, crop_len) in enumerate(audio_seg_gen):
            X = self.nsgt(audio_seg)
            Xmag = self.complexnorm(X)

            total += crop_len

            for j, (target_name, target_module) in enumerate(self.target_models.items()):
                Ymag = target_module(Xmag.detach().clone())
                Y = phasemix_sep(X, Ymag)
                y = self.insgt(Y)[..., : crop_len]

                estimates[j].append(y.detach().clone())

        for j, target_name in enumerate(self.target_models.keys()):
            estimates[j] = torch.cat(estimates[j], dim=-1)

        estimates = torch.cat([torch.unsqueeze(est, dim=0) for est in estimates], dim=0)

        # getting to (nb_samples, nb_targets, nb_channels, nb_samples)
        estimates = estimates.permute(1, 0, 2, 3).contiguous()

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
