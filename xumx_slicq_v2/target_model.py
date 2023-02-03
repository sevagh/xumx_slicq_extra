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

        channels = [nb_channels, 20, 40, 60]
        layers = len(channels) - 1

        if nb_f_bins < 10:
            freq_filter = 1
        elif nb_f_bins < 20:
            freq_filter = 3
        else:
            freq_filter = 7

        if nb_t_bins <= 100:
            time_filter = 7
        else:
            time_filter = 11

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
            ConvTranspose2d(nb_channels, nb_channels, (2, 7), stride=(1, 2), bias=True)
        )
        decoder.append(Sigmoid())

        cdae_1 = Sequential(*encoder, *decoder)
        cdae_2 = copy.deepcopy(cdae_1)
        cdae_3 = copy.deepcopy(cdae_1)
        cdae_4 = copy.deepcopy(cdae_1)

        self.cdaes = nn.ModuleList([cdae_1, cdae_2, cdae_3, cdae_4])
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

        ret = torch.zeros((4, *x_shape,), device=x.device, dtype=x.dtype)

        x = overlap_add_slicq(x)

        # shift and scale input to mean=0 std=1 (across all bins)
        x = x.permute(0, 1, 3, 2)
        x += self.input_mean
        x *= self.input_scale
        x = x.permute(0, 1, 3, 2)

        for i, cdae in enumerate(self.cdaes):
            x_tmp = x.clone()
            for j, layer in enumerate(cdae):
                x_tmp = layer(x_tmp)

            # crop
            x_tmp = x_tmp[:, :, : nb_f_bins, : nb_t_bins * nb_slices]

            x_tmp = x_tmp.reshape(x_shape)

            # multiplicative skip connection
            if self.mask:
                x_tmp = x_tmp * mix

            ret[i, ...] = x_tmp

        return ret


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
