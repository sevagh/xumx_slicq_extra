from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Linear, Parameter, ReLU, LSTM, GRU, Tanh, BatchNorm1d
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
        nb_layers (int): Number of Bi-LSTM layers (Default: `3`).
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
        nb_channels=2,
        nb_layers=3,
        hidden_size=512,
        temporal_encoding=32,
        unidirectional=False,
        input_mean=None,
        input_scale=None,
    ):
        super(OpenUnmix, self).__init__()

        self.nb_bins = nb_bins
        self.M = M
        self.temporal_encoding = temporal_encoding

        # temporal encoding
        self.fc1 = Linear(in_features=self.M, out_features=temporal_encoding, bias=False)
        self.act1 = Tanh()
        self.bn1 = BatchNorm1d(temporal_encoding)

        self.fc2 = Linear(in_features=temporal_encoding*nb_channels*nb_bins, out_features=hidden_size, bias=False)
        self.act2 = Tanh()
        self.bn2 = BatchNorm1d(hidden_size)

        self.hidden_size = hidden_size

        if unidirectional:
            rnn_hidden_size = hidden_size
        else:
            rnn_hidden_size = hidden_size // 2

        self.rnn = LSTM(
            input_size=hidden_size,
            hidden_size=rnn_hidden_size,
            num_layers=nb_layers,
            bidirectional=not unidirectional,
            batch_first=False,
            dropout=0.4 if nb_layers > 1 else 0,
        )

        fc3_hiddensize = hidden_size * 2
        self.fc3 = Linear(in_features=fc3_hiddensize, out_features=hidden_size, bias=False)
        self.act3 = ReLU()
        self.bn3 = BatchNorm1d(hidden_size)

        self.fc4 = Linear(in_features=hidden_size, out_features=temporal_encoding*nb_channels*nb_bins, bias=False)
        self.act4 = ReLU()
        self.bn4 = BatchNorm1d(temporal_encoding*nb_channels*nb_bins)

        # temporal decoding
        self.fc5 = Linear(in_features=temporal_encoding, out_features=self.M, bias=False)
        self.act5 = ReLU()

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

        nb_samples, nb_channels, nb_f_bins, nb_frames, nb_m_bins = x.shape

        x = x.reshape(nb_samples, nb_channels, nb_f_bins, -1)

        # permute so that batch is last for lstm
        x = x.permute(3, 0, 1, 2)

        # get collated nsgt spectrogram shape
        nb_frames_collated, nb_samples, nb_channels, nb_bins = x.data.shape

        # shift and scale input to mean=0 std=1 (across all bins)
        x = x + self.input_mean[: self.nb_bins]
        x = x * self.input_scale[: self.nb_bins]

        x_ = x.reshape(-1, self.M)

        # tanh for [-1,1]
        x = self.fc1(x.reshape(-1, self.M))
        x = self.act1(x)
        x = self.bn1(x)

        x = x.reshape(-1, self.temporal_encoding*nb_bins*nb_channels)
        #print(f'x.shape: {x.shape}')

        x = self.fc2(x)
        x = self.act2(x)
        x = self.bn2(x)

        # to (nb_frames*nb_samples, nb_channels*nb_bins)
        # and encode to (nb_frames*nb_samples, hidden_size)
        x = x.reshape(nb_frames, nb_samples, self.hidden_size)

        # apply 3-layers of stacked LSTM
        rnn_out, _ = self.rnn(x)

        # lstm skip connection
        x = torch.cat([x, rnn_out], -1)
        x = x.reshape(-1, x.shape[-1])

        # second dense stage + batch norm
        x = self.fc3(x)
        x = self.act3(x)
        x = self.bn3(x)

        # reshape back to original dim
        x = x.reshape(-1, self.hidden_size)

        x = self.fc4(x)
        x = self.act4(x)
        x = self.bn4(x)

        # decode with last linear layer
        x = self.fc5(x.reshape(-1, self.temporal_encoding))
        x = self.act5(x)

        x = x.reshape(nb_samples, nb_channels, nb_bins, -1, self.M)

        # relu because output is positive
        # multiply mix by learned mask above
        return x * mix


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
