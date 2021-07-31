from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Linear, Parameter, ReLU, Sigmoid, BatchNorm3d, Conv3d, ConvTranspose3d, Tanh, LSTM, BatchNorm1d
from .filtering import atan2
from .transforms import make_filterbanks, ComplexNorm, phasemix_sep, NSGTBase
from collections import defaultdict
import numpy as np
import logging

eps = 1.e-10


class OpenUnmix(nn.Module):
    """OpenUnmix Core spectrogram based separation module.

    Args:
        nb_bins, M (int): Number of sliCQ-NSGT tf bins
        nb_channels (int): Number of input audio channels (Default: `2`).
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
        unidirectional=False,
        nb_layers=1,
        input_mean=None,
        input_scale=None,
        info=False,
    ):
        super(OpenUnmix, self).__init__()

        self.nb_bins = nb_bins
        self.M = M

        # do 3D convolutions but don't touch the "slice" spatial dimension, which will be used for the GRU
        channels = [nb_channels, 12, 20, 30]
        filters = [(1, 3, 3), (1, 3, 3), (1, 3, 3)]
        strides = [(1, 3, 3), (1, 1, 1), (1, 3, 3)]
        dilations = [(1, 1, 1), (1, 1, 1), (1, 1, 1)]
        output_paddings = [(0, 1, 2), (0, 0, 0), (0, 2, 1)]

        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()

        layers = len(filters)

        self.conv_reduction = 10

        # input channel

        for i in range(layers):
            self.encoder.extend([
                Conv3d(channels[i], channels[i+1], filters[i], stride=strides[i], dilation=dilations[i], bias=False),
                BatchNorm3d(channels[i+1]),
                ReLU(), # tanh before lstm
            ])

        # 1x1 for dimensionality reduction
        #self.encoder.append(Conv3d(channels[-1], self.conv_reduction, 1))
        #self.encoder.append(BatchNorm3d(self.conv_reduction))
        #self.encoder.append(Tanh())

        # classic dense-RNN model of umx
        #hidden_size = 29*2*self.conv_reduction
        #self.hidden_size = hidden_size

        #if unidirectional:
        #    rnn_hidden_size = hidden_size
        #else:
        #    rnn_hidden_size = hidden_size // 2

        #self.rnn = LSTM(
        #    input_size=hidden_size,
        #    hidden_size=rnn_hidden_size,
        #    num_layers=nb_layers,
        #    bidirectional=not unidirectional,
        #    batch_first=False,
        #    #dropout=0.4 if nb_layers > 1 else 0,
        #)

        # 1x1 for dimensionality increase - 2 channels instead of 1 due to the skip connection
        #self.decoder.append(ConvTranspose3d(2*self.conv_reduction, channels[-1], 1))
        #self.decoder.append(BatchNorm3d(channels[-1]))
        #self.decoder.append(ReLU())

        for i in range(layers,1,-1):
            self.decoder.extend([
                ConvTranspose3d(channels[i], channels[i-1], filters[i-1], stride=strides[i-1], dilation=dilations[i-1], output_padding=output_paddings[i-1], bias=False),
                BatchNorm3d(channels[i-1]),
                ReLU(),
            ])

        # last layer has special activation
        self.decoder.append(ConvTranspose3d(channels[1], channels[0], filters[0], stride=strides[0], dilation=dilations[0], output_padding=output_paddings[0], bias=False))
        self.decoder.append(Sigmoid())
        self.mask = True

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

        self.info = info
        if self.info:
            logging.basicConfig(level=logging.INFO)

    def freeze(self):
        # set all parameters as not requiring gradient, more RAM-efficient
        # at test time
        for p in self.parameters():
            p.requires_grad = False
        self.eval()

    def forward(self, x: Tensor) -> Tensor:
        if self.info:
            print()

        mix = x.detach().clone()
        logging.info(f'0. mix {mix.shape}')

        nb_samples, nb_channels, nb_f_bins, nb_slices, nb_m_bins = x.shape

        x = x.reshape(nb_samples, nb_channels, nb_f_bins, -1)

        # permute so that batch is last for lstm
        x = x.permute(3, 0, 1, 2)

        logging.info(f'0. {x.shape}')

        # shift and scale input to mean=0 std=1 (across all bins)
        x = x + self.input_mean[: self.nb_bins]
        x = x * self.input_scale[: self.nb_bins]

        logging.info(f'1. POST-SCALE {x.shape}')

        x = x.reshape(nb_samples, nb_channels, nb_slices, nb_f_bins, nb_m_bins)

        logging.info(f'2. PRE-ENCODER {x.shape}')

        for i, layer in enumerate(self.encoder):
            sh1 = x.shape
            x = layer(x)
            sh2 = x.shape
            logging.info(f'\t2-{i} ENCODER {sh1} -> {sh2}')

        post_enc_shape = x.shape
        logging.info(f'3. POST-ENCODER {x.shape}')

        nb_samples, nb_conv_channels, _, nb_f_conv, nb_m_conv = x.shape

        # lstm layer

        # to (nb_frames*nb_samples, nb_channels*nb_bins)
        # and encode to (nb_frames*nb_samples, hidden_size)
        #x = x.reshape(nb_slices, nb_samples, self.hidden_size)

        #logging.info(f'4. PRE-RNN {x.shape}')

        ## apply 3-layers of stacked LSTM
        #rnn_out, _ = self.rnn(x)

        #logging.info(f'5. POST-RNN {rnn_out.shape}')

        ## lstm skip connection
        #x = torch.cat([x, rnn_out], -1)
        #x = x.reshape(-1, x.shape[-1])

        #logging.info(f'6. SKIP-CONN {x.shape}')

        ## reshape back to original dim, 2x chan due to the skip connection above
        #x = x.reshape(nb_samples, 2*nb_conv_channels, nb_slices, nb_f_conv, nb_m_conv)

        #logging.info(f'7. PRE-DECODER {x.shape}')

        for layer in self.decoder:
            sh1 = x.shape
            x = layer(x)
            sh2 = x.shape
            logging.info(f'\t7-{i} DECODER {sh1} -> {sh2}')

        logging.info(f'8. POST-DECODER {x.shape}')

        x = x.reshape(nb_samples, nb_channels, nb_slices, nb_f_bins, nb_m_bins)
        x = x.permute(0, 1, 3, 2, 4)

        logging.info(f'9. mix {mix.shape}')

        if self.mask:
            x = x*mix

        return x


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
        N = audio.shape[-1]

        estimates = torch.zeros(audio.shape + (nb_sources,), dtype=audio.dtype, device=self.device)

        for j, (target_name, target_module) in enumerate(self.target_models.items()):
            print(f'separating {target_name}')

            nsgt = self.nsgts[target_name]['nsgt']
            insgt = self.nsgts[target_name]['insgt']

            slicq_shape = nsgt.nsgt.predict_input_size(1, 2, self.seq_dur)
            seq_batch = slicq_shape[-2]

            X = nsgt(audio)
            Xmag = self.complexnorm(X)

            Xmagsegs = torch.split(Xmag, seq_batch, dim=3)
            Ymagsegs = []

            for Xmagseg in Xmagsegs:
                # apply current model to get the source magnitude spectrogram
                #Xmag_segs = torch.split(Xmag, 
                Ymagseg = target_module(Xmagseg.detach().clone())
                Ymagsegs.append(Ymagseg)

            Ymag = torch.cat(Ymagsegs, dim=3)

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
