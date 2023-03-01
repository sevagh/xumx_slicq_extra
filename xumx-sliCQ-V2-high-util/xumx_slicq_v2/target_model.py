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
from .transforms import (
    make_filterbanks,
    ComplexNorm,
    phasemix_sep,
    NSGTBase,
    overlap_add_slicq,
)
import copy

eps = 1.0e-10


class SlicedUnmix(nn.Module):
    def __init__(
        self,
        slicq_sample_input,
        input_mean=None,
        input_scale=None,
    ):
        super(SlicedUnmix, self).__init__()

        (
            nb_samples,
            nb_channels,
            nb_f_bins,
            nb_slices,
            nb_t_bins,
        ) = slicq_sample_input.shape

        channels = [nb_channels, 25, 55, 75]
        layers = len(channels) - 1

        if nb_f_bins < 10:
            freq_filter = 1
        elif nb_f_bins < 20:
            freq_filter = 3
        else:
            freq_filter = 7

        if nb_t_bins <= 100:
            time_filter = 9
        else:
            time_filter = 15

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
                    bias=False,
                )
            )
            decoder.append(BatchNorm2d(channels[i - 1]))
            decoder.append(ReLU())

        # grow the overlap-added half dimension to its full size
        decoder.append(
            ConvTranspose2d(nb_channels, nb_channels, (2, 7), stride=(1, 2), bias=True)
        )
        decoder.append(Sigmoid())

        self.cdae_1 = Sequential(*encoder, *decoder)
        self.cdae_2 = copy.deepcopy(self.cdae_1)
        self.cdae_3 = copy.deepcopy(self.cdae_1)
        self.cdae_4 = copy.deepcopy(self.cdae_1)

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
        x_masked = torch.zeros((4, *x_shape,), device=x.device, dtype=x.dtype)

        nb_samples, nb_channels, nb_f_bins, nb_slices, nb_t_bins = x_shape

        x = overlap_add_slicq(x)

        # shift and scale input to mean=0 std=1 (across all bins)
        x = x.permute(0, 1, 3, 2)
        x += self.input_mean
        x *= self.input_scale
        x = x.permute(0, 1, 3, 2)

        x1 = x
        x2 = x.clone()
        x3 = x.clone()
        x4 = x.clone()

        for i, layer in enumerate(self.cdae_1):
            x1 = layer(x1)
        for i, layer in enumerate(self.cdae_2):
            x2 = layer(x2)
        for i, layer in enumerate(self.cdae_3):
            x3 = layer(x3)
        for i, layer in enumerate(self.cdae_4):
            x4 = layer(x4)

        # crop
        x1 = x1[:, :, : nb_f_bins, : nb_t_bins * nb_slices]
        x2 = x2[:, :, : nb_f_bins, : nb_t_bins * nb_slices]
        x3 = x3[:, :, : nb_f_bins, : nb_t_bins * nb_slices]
        x4 = x4[:, :, : nb_f_bins, : nb_t_bins * nb_slices]

        x1 = x1.reshape(x_shape)
        x2 = x2.reshape(x_shape)
        x3 = x3.reshape(x_shape)
        x4 = x4.reshape(x_shape)

        # multiplicative skip connection
        x_masked[0, ...] = x1 * mix
        x_masked[1, ...] = x2 * mix
        x_masked[2, ...] = x3 * mix
        x_masked[3, ...] = x4 * mix

        return x_masked


class UnmixAllTargets(nn.Module):
    def __init__(
        self,
        jagged_slicq_sample_input,
        input_means=None,
        input_scales=None,
    ):
        super(UnmixAllTargets, self).__init__()

        self.sliced_umx = nn.ModuleList()

        for i, C_block in enumerate(jagged_slicq_sample_input):
            input_mean = input_means[i] if input_means else None
            input_scale = input_scales[i] if input_scales else None

            self.sliced_umx.append(
                SlicedUnmix(
                    C_block,
                    input_mean=input_mean,
                    input_scale=input_scale,
                )
            )

    def freeze(self):
        # set all parameters as not requiring gradient, more RAM-efficient
        # at test time
        for p in self.parameters():
            # p.requires_grad = False
            p.grad = None
        self.eval()

    def forward(self, Xmag) -> Tensor:
        futures = [
            torch.jit.fork(self.sliced_umx[i], Xmag_block)
            for i, Xmag_block in enumerate(Xmag)
        ]
        Ymag = [torch.jit.wait(future) for future in futures]
        return Ymag
