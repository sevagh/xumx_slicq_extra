from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter, ReLU, Tanh, BatchNorm1d, GRU, Linear
import itertools
from .filtering import atan2
from .transforms import make_filterbanks, ComplexNorm, phasemix_sep
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

        self.hidden_size = hidden_size

        if unidirectional:
            rnn_hidden_size = hidden_size
        else:
            rnn_hidden_size = hidden_size // 2

        self.fc1 = Linear(in_features=nb_channels*self.nb_bins, out_features=hidden_size)
        self.act1 = Tanh()
        self.bn1 = BatchNorm1d(hidden_size)

        self.rnn = GRU(
            input_size=hidden_size,
            hidden_size=rnn_hidden_size,
            num_layers=nb_layers,
            bidirectional=not unidirectional,
            batch_first=False,
            dropout=0.4 if nb_layers > 1 else 0,
        )

        fc2_hiddensize = hidden_size * 2
        self.fc2 = Linear(in_features=fc2_hiddensize, out_features=hidden_size, bias=False)
        self.act2 = ReLU()
        self.bn2 = BatchNorm1d(hidden_size)

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

        # permute into samples, frames, channels, f_bins, m_bins
        x = x.permute(3, 0, 1, 2)

        self.print(f'2. x.shape: {x.shape}')

        # get compressed nsgt spectrogram shape
        nb_frames, nb_samples, nb_channels, nb_bins = x.data.shape

        # shift and scale input to mean=0 std=1 (across all bins)
        x = x + self.input_mean[: self.nb_bins]
        x = x * self.input_scale[: self.nb_bins]

        self.print(f'3. x.shape: {x.shape}')

        # to (nb_frames*nb_samples, nb_channels*nb_bins)
        # and encode to (nb_frames*nb_samples, hidden_size)
        x = x.reshape(-1, nb_channels * nb_bins)
        x = self.fc1(x)

        self.print(f'4. x.shape: {x.shape}')

        # squash range to [-1, 1]
        x = self.act1(x)
        # normalize every instance in a batch
        x = self.bn1(x)
        x = x.reshape(nb_frames, nb_samples, self.hidden_size)

        self.print(f'5. x.shape: {x.shape}')

        # apply 3-layers of stacked GRU
        rnn_out, _ = self.rnn(x)

        self.print(f'6. rnn_out.shape: {rnn_out.shape}')

        # lstm skip connection
        x = torch.cat([x, rnn_out], -1)
        self.print(f'7. x.shape: {x.shape}')

        x = x.reshape(-1, x.shape[-1])
        self.print(f'8. x.shape: {x.shape}')

        # second dense stage + batch norm
        x = self.fc2(x)
        x = self.act2(x)
        x = self.bn2(x)

        self.print(f'9. x.shape: {x.shape}')

        # reshape back to original dim
        x = x.reshape(nb_samples, nb_channels, self.nb_bins, nb_frames)

        self.print(f'10. x.shape: {x.shape}')
        self.print(f'11. mix.shape: {mix.shape}')

        #if self.debug:
        #    input()

        return x * mix

    def print(self, msg):
        #if self.debug:
        #    print(msg)
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

        self.nsgts = defaultdict(dict)
        self.device = device

        # seq dur - make this customizeable
        # batch the seq dur make inference faster - multiple 6-second
        self.nsgt, self.insgt = make_filterbanks(
            6.0, sample_rate=sample_rate, device=device
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

        audio_segs = torch.split(audio, int(6*44100), dim=-1)

        estimates = [[]]*self.nb_targets

        for audio_seg in audio_segs:
            audio_seg_len = audio_seg.shape[-1]

            # pad if too short
            if audio_seg_len < 6*44100:
                audio_seg = torch.nn.functional.pad(audio_seg, (0, 6*44100-audio_seg_len), mode='constant', value=0)

            X = self.nsgt(audio_seg)
            Xmag = self.complexnorm(X)

            for j, (target_name, target_module) in enumerate(self.target_models.items()):
                #print(f'separating {target_name}')
                Ymag = target_module(Xmag.detach().clone())

                Y = phasemix_sep(X, Ymag)
                y = self.insgt(Y, audio_seg.shape[-1])

                # crop the padding
                if audio_seg_len < 6*44100:
                    y = y[:audio_seg_len]

                estimates[j].append(y)

        for j, target_name in enumerate(self.target_models.keys()):
            tmp = torch.cat([est for est in estimates[j]], dim=-1)
            print('tmp.shape: {0}'.format(tmp.shape))
            estimates[j] = tmp

        #print('estimates.shape: {0}'.format(estimates.shape))
        estimates = torch.cat([torch.unsqueeze(est, dim=0) for est in estimates], dim=0)
        print('estimates.shape: {0}'.format(estimates.shape))

        # getting to (nb_samples, nb_targets, nb_channels, nb_samples)
        #estimates = estimates.permute(0, 3, 1, 2).contiguous()
        estimates = torch.unsqueeze(estimates, dim=0).contiguous()
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
