from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Linear, Parameter, GRU, ReLU, Tanh, BatchNorm1d
from . import convolutional_rnn
from .filtering import atan2
from .transforms import make_filterbanks, ComplexNorm

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
        nb_bins=126,
        max_bin=116,
        M=304,
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

        self.M = M

        self.hidden_size = hidden_size

        if unidirectional:
            rnn_hidden_size = hidden_size
        else:
            rnn_hidden_size = hidden_size // 2

        self.fc1 = Linear(in_features=nb_channels*self.nb_bins*self.M, out_features=hidden_size, bias=False)
        self.bn1 = BatchNorm1d(hidden_size)
        self.act1 = Tanh()

        #self.rnn = convolutional_rnn.Conv2dGRU(
        #    in_channels=2,
        #    out_channels=1,
        #    num_layers=nb_layers,
        #    kernel_size=3,
        #    dilation=2,
        #    stride=2,
        #    bidirectional=not unidirectional,
        #    batch_first=True,
        #    dropout=0.4 if nb_layers > 1 else 0,
        #)

        self.rnn = GRU(
            input_size=hidden_size,
            hidden_size=rnn_hidden_size,
            #in_channels=2,
            #out_channels=1,
            num_layers=nb_layers,
            #kernel_size=3,
            #dilation=2,
            #stride=2,
            bidirectional=not unidirectional,
            batch_first=False,
            dropout=0.4 if nb_layers > 1 else 0,
        )

        #fc2_hiddensize = self.M * 2
        fc2_hiddensize = hidden_size * 2
        self.fc2 = Linear(in_features=fc2_hiddensize, out_features=hidden_size, bias=False)
        self.bn2 = BatchNorm1d(hidden_size)
        self.act2 = ReLU()

        self.fc3 = Linear(in_features=hidden_size, out_features=nb_channels*self.nb_output_bins*self.M, bias=False)
        self.bn3 = BatchNorm1d(nb_channels*self.nb_output_bins*self.M)
        self.act3 = ReLU()

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
        mix = x.detach().clone()
        #print('mix.shape: {0}'.format(mix.shape))

        # crop frequency bins
        x = x[:, :, : self.nb_bins, :, :]
        #print('x.shape: {0}'.format(x.shape))

        # combine slicq f x m coefficients
        x = x.reshape(*x.shape[:2], x.shape[2]*x.shape[-1], x.shape[-2])
        #print('x.shape: {0}'.format(x.shape))

        # permute so that batch is last for rnn
        x = x.permute(0, 3, 1, 2)
        #print('x.shape: {0}'.format(x.shape))

        #nb_samples, nb_frames, nb_channels, nb_f_bins, nb_m_bins = x.data.shape
        nb_samples, nb_frames, nb_channels, nb_bins = x.data.shape

        # shift and scale input to mean=0.5 std=1 (across all bins)
        x = x + self.input_mean[: self.nb_bins*self.M]
        x = x * self.input_scale[: self.nb_bins*self.M]

        # to (nb_frames*nb_samples, nb_channels*nb_bins)
        # and encode to (nb_frames*nb_samples, hidden_size)
        x = x.reshape(-1, nb_channels * nb_bins)
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.act1(x)

        #print('x.shape: {0}'.format(x.shape))

        # normalize every instance in a batch
        x = x.reshape(nb_frames, nb_samples, self.hidden_size)

        rnn_out = self.rnn(x)

        #print('x.shape: {0}'.format(x.shape))

        # skip conn
        x = torch.cat([x, rnn_out[0]], -1)
        x = x.reshape(-1, x.shape[-1])

        #print('x.shape: {0}'.format(x.shape))

        # second dense stage
        x = self.fc2(x)
        x = self.bn2(x)

        # relu activation because our output is positive
        x = self.act2(x)

        #print('x.shape: {0}'.format(x.shape))

        # third dense stage + batch norm
        x = self.fc3(x)
        x = self.bn3(x)

        # reshape back to original dim
        #print('x.shape: {0}'.format(x.shape))
        x = x.reshape(nb_frames, nb_samples, nb_channels, self.nb_output_bins*self.M)
        #print('x.shape: {0}'.format(x.shape))

        # permute back to (nb_samples, nb_channels, nb_bins, nb_frames)
        x = x.permute(1, 2, 3, 0)
        #x = x.permute(0, 2, 3, 1, 4)
        #print('x.shape: {0}'.format(x.shape))

        x = x.reshape(*x.shape[:2], self.nb_output_bins, x.shape[-1], self.M)

        #print('x.shape: {0}'.format(x.shape))
        #print('mix.shape: {0}'.format(mix.shape))

        # multiply mix by learned mask
        #return mask * mix
        return self.act3(x) * mix


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
        targets_mag_spectrogram = torch.zeros(X.shape + (nb_sources,), dtype=audio.dtype, device=X.device)

        for j, (target_name, target_module) in enumerate(self.target_models.items()):
            # apply current model to get the source magnitude spectrogram
            target_spectrogram = target_module(X.detach().clone())
            #targets_mag_spectrogram[..., j] = target_spectrogram

            #print(target_spectrogram)

            # multiply with mix phase and invert
            #mix_phase = atan2(mix_nsgt[..., 1], mix_nsgt[..., 0])

            #targets_nsgt[..., j, 0] = target_spectrogram*torch.cos(mix_phase)
            #targets_nsgt[..., j, 1] = target_spectrogram*torch.sin(mix_phase)
            targets_mag_spectrogram[..., j] = target_spectrogram

        y = (
            mix_nsgt[..., None]
            * (
                targets_mag_spectrogram
                / (eps + torch.sum(targets_mag_spectrogram, dim=-1, keepdim=True).to(mix_nsgt.dtype))
            )[..., None, :]
        )

        # getting to (nb_samples, nb_targets, channel, fft_size, n_frames, 2)
        y = y.permute(0, 6, 1, 2, 3, 4, 5).contiguous()

        print('y.shape: {0}'.format(y.shape))

        estimates = self.insgt(y, nb_samples)
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
