from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import LSTM, BatchNorm1d, Linear, Parameter, ReLU, Tanh, Sigmoid
from .filtering import atan2
from .transforms import make_filterbanks, ComplexNorm


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
        nb_bins=257,
        max_bin=237,
        M=292,
        max_m=None,
        nb_channels=2,
        hidden_size=512,
        nb_layers=3,
        unidirectional=False,
        input_mean=None,
        input_scale=None,
    ):
        super(OpenUnmix, self).__init__()

        self.nb_output_bins = nb_bins
        self.nb_bins = max_bin

        self.nb_output_M = M
        self.M = max_m if max_m else M

        self.hidden_size = hidden_size

        if unidirectional:
            lstm_hidden_size = hidden_size
        else:
            lstm_hidden_size = hidden_size // 2

        # reduce limited slicq coefficients to hidden size
        self.fc1 = Linear(in_features=nb_channels*self.nb_bins*self.M, out_features=hidden_size)
        self.bn1 = BatchNorm1d(hidden_size)
        self.tanh1 = Tanh()

        self.lstm = LSTM(
            input_size=hidden_size,
            hidden_size=lstm_hidden_size,
            num_layers=nb_layers,
            bidirectional=not unidirectional,
            batch_first=False,
            dropout=0.4 if nb_layers > 1 else 0,
        )

        fc2_hiddensize = hidden_size * 2
        self.fc2 = Linear(in_features=fc2_hiddensize, out_features=hidden_size, bias=False)
        self.bn2 = BatchNorm1d(hidden_size)
        self.relu2 = ReLU()

        self.fc3 = Linear(in_features=hidden_size, out_features=nb_channels*self.nb_output_bins*self.nb_output_M, bias=False)
        self.bn3 = BatchNorm1d(nb_channels*self.nb_output_bins*self.nb_output_M)
        #self.relu3 = ReLU()
        self.sig3 = Sigmoid()

        if input_mean is not None:
            input_mean = torch.from_numpy(-input_mean[: self.nb_bins]).float()
        else:
            input_mean = torch.zeros(self.nb_bins)

        if input_scale is not None:
            input_scale = torch.from_numpy(1.0 / input_scale[: self.nb_bins]).float()
        else:
            input_scale = torch.ones(self.nb_bins)

        self.input_mean = Parameter(input_mean)
        self.input_scale = Parameter(input_scale)

        self.output_scale = Parameter(torch.ones(nb_bins).float())
        self.output_mean = Parameter(torch.ones(nb_bins).float())

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

        #print('x.shape: {0}'.format(x.shape))

        # first, combine the M bins per each 126 frequency channel
        # into a single value, using simply the norm
        #x = torch.linalg.norm(x, dim=-1, ord=3)

        #print('x.shape: {0}'.format(x.shape))

        # crop frequency bins
        x = x[:, :, : self.nb_bins, :, : self.M]

        x = x.reshape(*x.shape[:2], x.shape[2]*x.shape[-1], x.shape[-2])

        #print('x.shape: {0}'.format(x.shape))

        # permute so that batch is last for lstm
        x = x.permute(3, 0, 1, 2)

        #print('x.shape: {0}'.format(x.shape))

        # get compressed nsgt spectrogram shape
        nb_frames, nb_samples, nb_channels, nb_bins = x.data.shape

        # shift and scale input to mean=0 std=1 (across all bins)
        #x = x + self.input_mean
        #x = x * self.input_scale

        # to (nb_frames*nb_samples, nb_channels*nb_bins)
        # and encode to (nb_frames*nb_samples, hidden_size)
        x = x.reshape(-1, nb_channels * nb_bins)
        x = self.fc1(x)

        # normalize every instance in a batch
        x = self.bn1(x)
        x = x.reshape(nb_frames, nb_samples, self.hidden_size)
        # squash range to [-1, 1]
        x = self.tanh1(x)

        # apply 3-layers of stacked LSTM
        lstm_out = self.lstm(x)

        # lstm skip connection
        x = torch.cat([x, lstm_out[0]], -1)
        x = x.reshape(-1, x.shape[-1])

        # second dense stage + batch norm
        x = self.fc2(x)
        x = self.bn2(x)

        x = self.relu2(x)

        # third dense stage + batch norm
        x = self.fc3(x)
        x = self.bn3(x)

        # reshape back to original dim
        x = x.reshape(nb_frames, nb_samples, nb_channels, self.nb_output_bins*self.nb_output_M)

        #print('x.shape: {0}'.format(x.shape))

        # apply output scaling
        #x *= self.output_scale
        #x += self.output_mean

        #print('mix.shape: {0}'.format(mix.shape))

        # permute back to (nb_samples, nb_channels, nb_bins, nb_frames)
        x = x.permute(1, 2, 3, 0)

        #print('x.shape: {0}'.format(x.shape))

        x = x.reshape(*x.shape[:2], self.nb_output_bins, x.shape[-1], self.nb_output_M)

        # since our output is non-negative, we can apply ReLU
        #x = self.relu3(x)

        # reintroduce M time bins
        #x = torch.unsqueeze(x, dim=-1)
        #x = x.expand(*x.shape[:4], self.M)

        # try to learn the final ReLU across all M time bins
        # for a more finely-grained mask

        # multiply mix by learned mask above
        return self.sig3(x) * mix


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
        niter: int = 0,
        softmask: bool = False,
        residual: bool = False,
        sample_rate: float = 44100.0,
        nb_channels: int = 2,
        wiener_win_len: Optional[int] = 300,
        device: str = "cpu",
    ):
        super(Separator, self).__init__()

        # saving parameters
        self.niter = niter
        self.residual = residual
        self.softmask = softmask
        self.wiener_win_len = wiener_win_len

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
        mix_nsgt = self.nsgt(audio)
        X = self.complexnorm(mix_nsgt)

        # initializing spectrograms variable
        #targets_mag_spectrogram = torch.zeros(X.shape + (nb_sources,), dtype=audio.dtype, device=X.device)
        targets_nsgt = torch.zeros(X.shape + (nb_sources,2), dtype=audio.dtype, device=X.device)

        for j, (target_name, target_module) in enumerate(self.target_models.items()):
            # apply current model to get the source magnitude spectrogram
            target_spectrogram = target_module(X.detach().clone())
            #targets_mag_spectrogram[..., j] = target_spectrogram

            # multiply with mix phase and invert
            mix_phase = atan2(mix_nsgt[..., 1], mix_nsgt[..., 0])

            targets_nsgt[..., j, 0] = target_spectrogram*torch.cos(mix_phase)
            targets_nsgt[..., j, 1] = target_spectrogram*torch.sin(mix_phase)

        # getting to (nb_samples, nb_targets, channel, fft_size, n_frames, 2)
        targets_nsgt = targets_nsgt.permute(0, 5, 1, 2, 3, 4, 6).contiguous()

        estimates = self.insgt(targets_nsgt, nb_samples)
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

        # in the case of residual, we added another source
        if self.residual:
            estimates_dict["residual"] = estimates[:, -1, ...]

        if aggregate_dict is not None:
            new_estimates = {}
            for key in aggregate_dict:
                new_estimates[key] = torch.tensor(0.0)
                for target in aggregate_dict[key]:
                    new_estimates[key] = new_estimates[key] + estimates_dict[target]
            estimates_dict = new_estimates
        return estimates_dict
