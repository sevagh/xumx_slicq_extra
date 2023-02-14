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
from .transforms import (
    make_filterbanks,
    phasemix_sep,
    ComplexNorm,
    NSGTBase,
)
import copy


class Unmix(nn.Module):
    def __init__(
        self,
        jagged_slicq_sample_input,
        encoder,
        max_bin=None,
        input_means=None,
        input_scales=None,
    ):
        super(Unmix, self).__init__()

        self.nsgt, self.insgt, self.cnorm = encoder
        self.umx = _UnmixAllTargets(jagged_slicq_sample_input, max_bin, input_means, input_scales)

    def freeze(self):
        # set all parameters as not requiring gradient, more RAM-efficient
        # at test time
        for p in self.parameters():
            # p.requires_grad = False
            p.grad = None
        self.eval()

    def forward(self, x, return_nsgts: bool = False) -> Tensor:
        n_samples = x.shape[-1]

        X= self.nsgt(x)
        Xmag = self.cnorm(X)

        Ymag_all = self.umx(Xmag)

        Ycomplex_all = phasemix_sep(X, Ymag_all)

        y_all = self.insgt(Ycomplex_all, n_samples)
        if return_nsgts:
            return y_all, Ycomplex_all
        return y_all


# inner class for doing umx for all targets for all slicqt blocks
class _UnmixAllTargets(nn.Module):
    def __init__(
        self,
        jagged_slicq_sample_input,
        max_bin=None,
        input_means=None,
        input_scales=None,
    ):
        super(_UnmixAllTargets, self).__init__()

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

    def forward(self, X) -> Tensor:
        futures = [
            torch.jit.fork(self.sliced_umx[i], X_block)
            for i, X_block in enumerate(X)
        ]
        Ycomplex = [torch.jit.wait(future) for future in futures]
        return Ycomplex


# inner class for doing umx for all targets per slicqt block
class _SlicedUnmix(nn.Module):
    def __init__(
        self,
        slicq_sample_input,
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

        encoder = []
        decoder = []

        window = nb_t_bins
        hop = window // 2
        self.ncoefs = nb_slices * nb_t_bins // 2 + hop

        encoder.extend([
            Conv2d(
                nb_channels,
                55,
                (freq_filter, window),
                stride=(1, hop),
                bias=False,
            ),
            BatchNorm2d(55),
            ReLU(),
        ])

        encoder.extend([
            Conv2d(
                55,
                75,
                (freq_filter, 3),
                bias=False,
            ),
            BatchNorm2d(75),
            ReLU(),
        ])

        decoder.extend([
            ConvTranspose2d(
                75,
                55,
                (freq_filter, 3),
                bias=False,
            ),
            BatchNorm2d(55),
            ReLU(),
        ])

        decoder.extend([
            ConvTranspose2d(
                55,
                nb_channels,
                (freq_filter, window),
                stride=(1, hop),
                bias=True,
            ),
            Sigmoid(),
        ])

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

        x = torch.flatten(x, start_dim=-2, end_dim=-1)

        # shift and scale input to mean=0 std=1 (across all bins)
        x = x.permute(0, 1, 3, 2)
        x += self.input_mean
        x *= self.input_scale
        x = x.permute(0, 1, 3, 2)

        for i, cdae in enumerate(self.cdaes):
            x_tmp = x.clone()
            #print(f"x: {x_tmp.shape}")
            for j, layer in enumerate(cdae):
                x_tmp = layer(x_tmp)
                #print(f"\tx {j}: {x_tmp.shape}")

            #print(f"overlap would have: {self.ncoefs}")
            #import sys; sys.exit(0)

            # crop
            #x_tmp = x_tmp[:, :, : nb_f_bins, : nb_t_bins * nb_slices]

            x_tmp = x_tmp.reshape(x_shape)

            # multiplicative skip connection
            if self.mask:
                x_tmp = x_tmp * mix

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
