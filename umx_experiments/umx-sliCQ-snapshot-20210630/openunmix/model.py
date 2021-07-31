from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter, ReLU, Linear, LSTM, Tanh, BatchNorm1d
from .filtering import atan2, wiener
from .transforms import make_filterbanks, ComplexNorm, phasemix_sep, NSGTBase, overlap_add_slicq
from collections import defaultdict
import numpy as np
import logging

eps = 1.e-10


# just pass input through directly
class DummyTimeBucket(nn.Module):
    def __init__(
        self,
        slicq_sample_input,
        unidirectional=False,
        info=False,
    ):
        super(DummyTimeBucket, self).__init__()

    def freeze(self):
        # set all parameters as not requiring gradient, more RAM-efficient
        # at test time
        for p in self.parameters():
            p.requires_grad = False
        self.eval()

    def forward(self, x: Tensor) -> Tensor:
        mix = x.detach().clone()
        return mix


class OpenUnmixTimeBucket(nn.Module):
    def __init__(
        self,
        slicq_sample_input,
        unidirectional=False,
        info=False,
    ):
        super(OpenUnmixTimeBucket, self).__init__()

        nb_samples, nb_channels, nb_f_bins, nb_slices, nb_t_bins = slicq_sample_input.shape

        self.nb_bins = nb_f_bins
        hidden_size = nb_f_bins*nb_channels
        self.hidden_size = hidden_size

        self.bn1 = BatchNorm1d(self.hidden_size)

        if unidirectional:
            lstm_hidden_size = hidden_size
        else:
            lstm_hidden_size = hidden_size // 2

        self.lstm = LSTM(
            input_size=hidden_size,
            hidden_size=lstm_hidden_size,
            num_layers=2,
            bidirectional=not unidirectional,
            batch_first=False,
            dropout=0.4,
        )

        fc2_hiddensize = hidden_size * 2
        self.fc2 = Linear(in_features=fc2_hiddensize, out_features=hidden_size, bias=False)

        self.bn2 = BatchNorm1d(hidden_size)

    def freeze(self):
        # set all parameters as not requiring gradient, more RAM-efficient
        # at test time
        for p in self.parameters():
            p.requires_grad = False
        self.eval()

    def forward(self, x: Tensor) -> Tensor:
        mix = x.detach().clone()
        logging.info(f'0. mix shape: {mix.shape}')
        logging.info(f'0. x shape: {x.shape}')

        x_shape = x.shape
        nb_samples, nb_channels, nb_f_bins, nb_slices, nb_t_bins = x_shape

        # stack slice with batch
        x = x.reshape(nb_samples*nb_slices, nb_channels, nb_f_bins, nb_t_bins)

        # to (nb_frames*nb_samples, nb_channels*nb_bins)
        # and encode to (nb_frames*nb_samples, hidden_size)
        x = x.reshape(-1, nb_channels * self.nb_bins)
        # normalize every instance in a batch
        x = self.bn1(x)
        x = x.reshape(nb_t_bins, nb_samples*nb_slices, self.hidden_size)
        # squash range ot [-1, 1]
        x = torch.tanh(x)

        # apply 3-layers of stacked LSTM
        lstm_out = self.lstm(x)

        # lstm skip connection
        x = torch.cat([x, lstm_out[0]], -1)

        # first dense stage + batch norm
        x = self.fc2(x.reshape(-1, x.shape[-1]))
        x = self.bn2(x)

        #x = F.relu(x)

        # second dense stage + layer norm
        #x = self.fc3(x)
        #x = self.bn3(x)

        # reshape back to original dim
        x = x.reshape(x_shape)

        # since our output is non-negative, we can apply RELU
        x = F.relu(x) * mix
        # permute back to (nb_samples, nb_channels, nb_bins, nb_frames)
        return x


class OpenUnmix(nn.Module):
    """OpenUnmix Core spectrogram based separation module.

    Args:
        nb_bins, M (int): Number of sliCQ-NSGT tf bins
        nb_channels (int): Number of input audio channels (Default: `2`).
    """

    def __init__(
        self,
        jagged_slicq_sample_input,
        max_bin=None,
        unidirectional=False,
        info=False,
    ):
        super(OpenUnmix, self).__init__()

        self.bucketed_unmixes = nn.ModuleList()

        freq_idx = 0
        for i, C_block in enumerate(jagged_slicq_sample_input):
            freq_start = freq_idx

            if max_bin is not None and freq_start >= max_bin:
                self.bucketed_unmixes.append(DummyTimeBucket(C_block))
            else:
                self.bucketed_unmixes.append(OpenUnmixTimeBucket(C_block))

            # advance global frequency pointer
            freq_idx += C_block.shape[2]

        self.info = info
        if self.info:
            logging.basicConfig(level=logging.INFO)

    def freeze(self):
        # set all parameters as not requiring gradient, more RAM-efficient
        # at test time
        for p in self.parameters():
            p.requires_grad = False
        self.eval()

    def forward(self, x) -> Tensor:
        futures = [torch.jit.fork(self.bucketed_unmixes[i], Xmag_block) for i, Xmag_block in enumerate(x)]
        y = [torch.jit.wait(future) for future in futures]
        return y


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
    """
    def __init__(
        self,
        target_models: dict,
        target_models_nsgt: dict,
        niter: int = 0,
        sample_rate: float = 44100.0,
        single_nsgt: bool = True,
        nb_channels: int = 2,
        device: str = "cpu",
        softmask: bool = False,
    ):
        super(Separator, self).__init__()

        # saving parameters
        self.niter = niter
        self.softmask = softmask
        self.single_nsgt = single_nsgt

        self.device = device

        if self.single_nsgt:
            nsgt_base = next(iter(target_models_nsgt.values()))
            self.nsgt, self.insgt = make_filterbanks(
                nsgt_base, sample_rate=sample_rate
            )
        else:
            self.nsgts = defaultdict(dict)
            # separate nsgt per model
            for name, nsgt_base in target_models_nsgt.items():
                nsgt, insgt = make_filterbanks(
                    nsgt_base, sample_rate=sample_rate
                )

                self.nsgts[name]['nsgt'] = nsgt
                self.nsgts[name]['insgt'] = insgt

        self.complexnorm = ComplexNorm(mono=nb_channels == 1)
        self.nb_channels = nb_channels

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

        estimates = torch.zeros(audio.shape + (nb_sources,), dtype=audio.dtype, device=self.device)

        if not self.single_nsgt:
            for j, (target_name, target_module) in enumerate(self.target_models.items()):
                nsgt = self.nsgts[target_name]['nsgt']
                insgt = self.nsgts[target_name]['insgt']

                X = nsgt(audio)
                Xmag = self.complexnorm(X)

                Ymag = target_module(Xmag)

                Ycomplex = [None]*len(X)
                for i, X_block in enumerate(X):
                    Ycomplex[i] = phasemix_sep(X_block, Ymag[i])

                y = insgt(Ycomplex, audio.shape[-1])
                estimates[..., j] = y
        else:
            X = self.nsgt(audio)
            Xmag = self.complexnorm(X)

            # initializing spectrograms variable
            Ymags = [None]*nb_sources

            for j, (target_name, target_module) in enumerate(self.target_models.items()):
                # apply current model to get the source spectrogram
                Ymags[j] = target_module(Xmag)

            if nb_sources == 1 and self.niter > 0:
                raise Exception(
                    "Cannot use EM if only one target is estimated."
                    "Provide two targets or create an additional "
                    "one with `--residual`"
                )

            Y = [None]*len(X)

            # slice-wise wiener, no win len or pos
            for i, X_block in enumerate(X):
                # iterate over target list and concatenate it into a (,targets) tensor
                tmp = torch.cat([torch.unsqueeze(Ymag_block, dim=-1) for Ymag_block in Ymags[i]], dim=-1)
                Y[i] = wiener(
                    tmp,
                    X_block,
                    self.niter,
                    softmask=self.softmask,
                    residual=False,
                )

            est_lists = [[None]*len(X)]*nb_sources

            for i, Y_block in enumerate(Y):
                for j in range(nb_sources):
                    est_lists[j][i] = Y_block[..., j].contiguous()

            for j in range(nb_sources):
                estimates[..., j] = self.insgt(est_lists[j], length=audio.shape[2])

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
