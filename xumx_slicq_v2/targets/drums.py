from typing import Optional

from tqdm import trange
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import (
    Parameter,
    ReLU,
    BatchNorm2d,
    ConvTranspose2d,
    Conv2d,
    Sequential,
    Sigmoid,
)
import norbert
from ..transforms import (
    make_filterbanks,
    ComplexNorm,
    phasemix_sep,
    NSGTBase,
    overlap_add_slicq,
)
import numpy as np
import copy

eps = 1.0e-10


class SlicedUnmixDrums(nn.Module):
    def __init__(
        self,
        slicq_sample_input,
        input_mean=None,
        input_scale=None,
    ):
        super(SlicedUnmixDrums, self).__init__()

        (
            nb_samples,
            nb_channels,
            nb_f_bins,
            nb_slices,
            nb_t_bins,
        ) = slicq_sample_input.shape

        channels = [nb_channels, 25, 55]
        layers = len(channels) - 1

        if nb_f_bins < 10:
            freq_filter = 1
        elif nb_f_bins < 20:
            freq_filter = 3
        else:
            freq_filter = 5

        if nb_t_bins <= 100:
            time_filter = 7
        else:
            time_filter = 13

        filters = [(freq_filter, time_filter)] * layers

        encoder = []
        decoder = []

        layers = len(filters)

        for i in range(layers):
            encoder.append(
                Conv2d(
                    channels[i],
                    channels[i + 1],
                    filters[i],
                    dilation=(1, 2),
                    bias=False,
                )
            )
            encoder.append(
                BatchNorm2d(channels[i + 1]),
            )
            encoder.append(
                ReLU(),
            )

        for i in range(layers, 0, -1):
            decoder.append(
                ConvTranspose2d(
                    channels[i],
                    channels[i - 1],
                    filters[i - 1],
                    dilation=(1, 2),
                    bias=False,
                )
            )
            decoder.append(BatchNorm2d(channels[i - 1]))
            decoder.append(ReLU())

        # grow the overlap-added half dimension to its full size
        decoder.append(
            ConvTranspose2d(nb_channels, nb_channels, (1, 3), stride=(1, 2), bias=True)
        )
        decoder.append(Sigmoid())

        self.cdae = Sequential(*encoder, *decoder)
        self.mask = True

        if input_mean is not None:
            input_mean = torch.from_numpy(-input_mean).float()
        else:
            input_mean = torch.zeros(nb_f_bins)

        if input_scale is not None:
            input_scale = torch.from_numpy(1.0 / input_scale).float()
        else:
            input_scale = torch.ones(nb_f_bins)

        self.input_mean = Parameter(input_mean)
        self.input_scale = Parameter(input_scale)

    def freeze(self):
        # set all parameters as not requiring gradient, more RAM-efficient
        # at test time
        for p in self.parameters():
            # p.requires_grad = False
            p.grad = None
        self.eval()

    def forward(self, x: Tensor) -> Tensor:
        mix = x.detach().clone()

        x_shape = x.shape
        nb_samples, nb_channels, nb_f_bins, nb_slices, nb_t_bins = x_shape

        x = overlap_add_slicq(x)

        # shift and scale input to mean=0 std=1 (across all bins)
        x = x.permute(0, 1, 3, 2)
        x += self.input_mean
        x *= self.input_scale
        x = x.permute(0, 1, 3, 2)

        for i, layer in enumerate(self.cdae):
            x = layer(x)

        # crop
        x = x[:, :, :, : nb_t_bins * nb_slices]

        x = x.reshape(x_shape)

        # multiplicative skip connection
        if self.mask:
            x = x * mix

        return x


class UnmixDrums(nn.Module):
    def __init__(
        self,
        jagged_slicq_sample_input,
        input_means=None,
        input_scales=None,
    ):
        super(UnmixDrums, self).__init__()

        self.sliced_umx_drums = nn.ModuleList()

        freq_idx = 0
        for i, C_block in enumerate(jagged_slicq_sample_input):
            input_mean = input_means[i] if input_means else None
            input_scale = input_scales[i] if input_scales else None

            freq_start = freq_idx

            self.sliced_umx_drums.append(
                SlicedUnmixDrums(
                    C_block,
                    input_mean=input_mean,
                    input_scale=input_scale,
                )
            )

            # advance global frequency pointer
            freq_idx += C_block.shape[2]

    def freeze(self):
        # set all parameters as not requiring gradient, more RAM-efficient
        # at test time
        for p in self.parameters():
            # p.requires_grad = False
            p.grad = None
        self.eval()

    def forward(self, x) -> Tensor:
        futures_drums = [
            torch.jit.fork(self.sliced_umx_drums[i], Xmag_block)
            for i, Xmag_block in enumerate(x)
        ]
        y_drums = [torch.jit.wait(future) for future in futures_drums]
        return y_drums
