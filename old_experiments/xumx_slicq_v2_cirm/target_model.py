from typing import Optional

from tqdm import trange
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import (
    Parameter,
    Tanh,
    ConvTranspose2d,
    Conv2d,
    Sequential,
)
import norbert
from .transforms import (
    make_filterbanks,
    NSGTBase,
    overlap_add_slicq,
)
import copy

eps = 1.0e-10


# just pass input through directly
class DummyTimeBucket(nn.Module):
    def __init__(
        self,
        slicq_sample_input,
    ):
        super(DummyTimeBucket, self).__init__()

    def freeze(self):
        # set all parameters as not requiring gradient, more RAM-efficient
        # at test time
        for p in self.parameters():
            #p.requires_grad = False
            p.grad = None
        self.eval()

    def forward(self, x: Tensor) -> Tensor:
        return torch.unsqueeze(x, dim=0).repeat(4, 1, 1, 1, 1, 1)


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
                    bias=True,
                    dtype=torch.complex64,
                )
            )
            encoder.append(
                Tanh(),
            )

        for i in range(layers, 0, -1):
            decoder.append(
                ConvTranspose2d(
                    channels[i],
                    channels[i - 1],
                    filters[i - 1],
                    dilation=(1, 2),
                    bias=True,
                    dtype=torch.complex64,
                )
            )
            decoder.append(Tanh())

        # grow the overlap-added half dimension to its full size
        decoder.append(
            ConvTranspose2d(nb_channels, nb_channels, (1, 3), stride=(1, 2), bias=True, dtype=torch.complex64)
        )

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
        max_bin=None,
        input_means=None,
        input_scales=None,
    ):
        super(UnmixAllTargets, self).__init__()

        self.sliced_umx = nn.ModuleList()

        freq_idx = 0
        for i, C_block in enumerate(jagged_slicq_sample_input):
            input_mean = input_means[i] if input_means else None
            input_scale = input_scales[i] if input_scales else None

            freq_start = freq_idx

            if max_bin is not None and freq_start >= max_bin:
                self.sliced_umx.append(
                    DummyTimeBucket(C_block)
                )
            else:
                self.sliced_umx.append(
                    SlicedUnmix(
                        C_block,
                        input_mean=input_mean,
                        input_scale=input_scale,
                    )
                )

            # advance frequency pointer
            freq_idx += C_block.shape[2]

    def freeze(self):
        # set all parameters as not requiring gradient, more RAM-efficient
        # at test time
        for p in self.parameters():
            # p.requires_grad = False
            p.grad = None
        self.eval()

    def forward(self, X) -> Tensor:
        futures = [
            torch.jit.fork(self.sliced_umx[i], X_block)
            for i, X_block in enumerate(X)
        ]
        Ymag = [torch.jit.wait(future) for future in futures]
        return Ymag
