from typing import Optional
from tqdm import trange
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import (
    Parameter,
    BatchNorm2d,
    ConvTranspose2d,
    Conv2d,
    Sequential,
    ReLU,
    Sigmoid,
)
from .transforms import (
    make_filterbanks,
    phasemix_sep,
    ComplexNorm,
    NSGTBase,
)
import copy


# inner class for doing umx for all targets for all slicqt blocks
class Unmix(nn.Module):
    def __init__(
        self,
        jagged_slicq_sample_input,
        max_bin=None,
        input_means=None,
        input_scales=None,
    ):
        super(Unmix, self).__init__()

        self.sliced_umx = nn.ModuleList()

        freq_idx = 0
        for i, C_block in enumerate(jagged_slicq_sample_input):
            input_mean = input_means[i] if input_means else None
            input_scale = input_scales[i] if input_scales else None

            freq_start = freq_idx

            if max_bin is not None and freq_start >= max_bin:
                self.sliced_umx.append(
                    _DummyTimeBucket(C_block)
                )
            else:
                self.sliced_umx.append(
                    _SlicedUnmix(
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

    def forward(self, Xmag) -> Tensor:
        futures = [
            torch.jit.fork(self.sliced_umx[i], Xmag_block)
            for i, Xmag_block in enumerate(Xmag)
        ]
        Ymag = [torch.jit.wait(future) for future in futures]
        return Ymag


# inner class for doing umx for all targets per slicqt block
class _SlicedUnmix(nn.Module):
    def __init__(
        self,
        slicq_sample_input,
        hidden_size_1: int = 96,
        hidden_size_2: int = 184,
        input_mean=None,
        input_scale=None,
    ):
        super(_SlicedUnmix, self).__init__()

        (
            nb_samples,
            nb_channels,
            nb_f_bins,
            nb_slices,
            nb_t_bins,
        ) = slicq_sample_input.shape

        if nb_f_bins < 10:
            freq_filter = 1
        elif nb_f_bins < 20:
            freq_filter = 3
        else:
            freq_filter = 5

        window = nb_t_bins
        hop = window // 2
        self.ncoefs = nb_slices * nb_t_bins // 2 + hop

        encoder_1 = nn.Sequential(*[
            Conv2d(
                nb_channels,
                hidden_size_1,
                (freq_filter, window),
                stride=(1, hop),
                bias=False,
            ),
            BatchNorm2d(hidden_size_1),
            ReLU(),
        ])

        encoder_2 = nn.Sequential(*[
            Conv2d(
                hidden_size_1,
                hidden_size_2,
                (freq_filter, 3),
                bias=False,
            ),
            BatchNorm2d(hidden_size_2),
            ReLU(),
        ])

        decoder_1 = nn.Sequential(*[
            ConvTranspose2d(
                hidden_size_2,
                hidden_size_1,
                (freq_filter, 3),
                bias=False,
            ),
            BatchNorm2d(hidden_size_1),
            ReLU(),
        ])

        decoder_2 = nn.Sequential(*[
            ConvTranspose2d(
                hidden_size_1,
                nb_channels,
                (freq_filter, window),
                stride=(1, hop),
                bias=True,
            ),
            Sigmoid()
        ])

        self.encoder1s = nn.ModuleList([encoder_1, copy.deepcopy(encoder_1), copy.deepcopy(encoder_1), copy.deepcopy(encoder_1)])
        self.encoder2s = nn.ModuleList([encoder_2, copy.deepcopy(encoder_2), copy.deepcopy(encoder_2), copy.deepcopy(encoder_2)])
        self.decoder1s = nn.ModuleList([decoder_1, copy.deepcopy(decoder_1), copy.deepcopy(decoder_1), copy.deepcopy(decoder_1)])
        self.decoder2s = nn.ModuleList([decoder_2, copy.deepcopy(decoder_2), copy.deepcopy(decoder_2), copy.deepcopy(decoder_2)])

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
        # first skip conn, i.e. the multiplicative skip connection or soft-mask
        x_skip_1 = x.detach().clone()

        x_shape = x.shape
        nb_samples, nb_channels, nb_f_bins, nb_slices, nb_t_bins = x_shape
        #print(f"{nb_slices=}, {nb_t_bins=}")

        ret = torch.zeros((4, *x_shape,), device=x.device, dtype=x.dtype)

        x = torch.flatten(x, start_dim=-2, end_dim=-1)

        # shift and scale input to mean=0 std=1 (across all bins)
        x = x.permute(0, 1, 3, 2)
        x += self.input_mean
        x *= self.input_scale
        x = x.permute(0, 1, 3, 2)

        # operate per-target
        for i in range(4):
            # apply encoder1->encoder2->decoder1->decoder2
            # with skip connection; one from encoder1->decoder1, one from input to output

            x_tmp = x.clone()

            # apply first encoder layer
            x_tmp = self.encoder1s[i](x_tmp)

            # second skip connection
            #x_skip_2 = x_tmp.clone()

            #print(f"encoder 1: {x_tmp.shape}")
            #print(f"skip 2: {x_tmp.shape}")

            # apply second encoder layer
            x_tmp = self.encoder2s[i](x_tmp)

            #print(f"encoder 2: {x_tmp.shape}")

            # apply first decoder layer
            x_tmp = self.decoder1s[i](x_tmp)

            #print(f"decoder 1: {x_tmp.shape}")

            # add second skip connection
            #x_tmp = x_tmp + x_skip_2

            # apply second decoder layer after second skip conn
            x_tmp = self.decoder2s[i](x_tmp)

            # multiplicative skip connection of original mix
            x_tmp = x_tmp.reshape(x_shape)
            x_tmp = x_tmp * x_skip_1

            ret[i, ...] = x_tmp

        return ret


# just pass input through directly for frequency bins above bandwidth
class _DummyTimeBucket(nn.Module):
    def __init__(
        self,
        slicq_sample_input,
    ):
        super(_DummyTimeBucket, self).__init__()

    def freeze(self):
        # set all parameters as not requiring gradient, more RAM-efficient
        # at test time
        for p in self.parameters():
            #p.requires_grad = False
            p.grad = None
        self.eval()

    def forward(self, x: Tensor) -> Tensor:
        return torch.unsqueeze(x, dim=0).repeat(4, 1, 1, 1, 1, 1)
