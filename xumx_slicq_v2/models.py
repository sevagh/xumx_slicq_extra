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
    NSGTBase,
)
import copy


# inner class for doing umx for all targets for all slicqt blocks
class Unmix(nn.Module):
    def __init__(
        self,
        jagged_slicq_sample_input,
        input_means_mag=None,
        input_scales_mag=None,
        input_means_phase=None,
        input_scales_phase=None,
    ):
        super(Unmix, self).__init__()

        self.sliced_umx_mag = nn.ModuleList()
        self.sliced_umx_phase = nn.ModuleList()

        freq_idx = 0
        for i, C_block in enumerate(jagged_slicq_sample_input):
            input_mean_mag = input_means_mag[i] if input_means_mag else None
            input_scale_mag = input_scales_mag[i] if input_scales_mag else None

            input_mean_phase = input_means_phase[i] if input_means_phase else None
            input_scale_phase = input_scales_phase[i] if input_scales_phase else None

            freq_start = freq_idx

            self.sliced_umx_mag.append(
                _SlicedUnmix(
                    C_block,
                    input_mean=input_mean_mag,
                    input_scale=input_scale_mag,
                )
            )
            self.sliced_umx_phase.append(
                _SlicedUnmix(
                    C_block,
                    input_mean=input_mean_phase,
                    input_scale=input_scale_phase,
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

    def forward(self, Xcomplex) -> Tensor:
        # magnitude with torch.abs
        futures_mag = [
            torch.jit.fork(self.sliced_umx_mag[i], torch.abs(Xblock))
            for i, Xblock in enumerate(Xcomplex)
        ]
        # phase with torch.angle
        futures_phase = [
            torch.jit.fork(self.sliced_umx_phase[i], torch.angle(Xblock))
            for i, Xblock in enumerate(Xcomplex)
        ]
        Ymag = [torch.jit.wait(future) for future in futures_mag]
        Yphase = [torch.jit.wait(future) for future in futures_phase]
        Ycomplex = [torch.polar(Ymag_block, Yphase_block) for (Ymag_block, Yphase_block) in zip(Ymag, Yphase)]
        return Ycomplex


# inner class for doing umx for all targets per slicqt block
class _SlicedUnmix(nn.Module):
    def __init__(
        self,
        slicq_sample_input,
        input_mean=None,
        input_scale=None,
    ):
        hidden_size_1 = 25
        hidden_size_2 = 55

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
                hidden_size_1,
                (freq_filter, window),
                stride=(1, hop),
                bias=False,
            ),
            BatchNorm2d(hidden_size_1),
            ReLU(),
        ])

        encoder.extend([
            Conv2d(
                hidden_size_1,
                hidden_size_2,
                (freq_filter, 3),
                bias=False,
            ),
            BatchNorm2d(hidden_size_2),
            ReLU(),
        ])

        decoder.extend([
            ConvTranspose2d(
                hidden_size_2,
                hidden_size_1,
                (freq_filter, 3),
                bias=False,
            ),
            BatchNorm2d(hidden_size_1),
            ReLU(),
        ])

        decoder.extend([
            ConvTranspose2d(
                hidden_size_1,
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

    # x is either a magnitude or phase
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
            for j, layer in enumerate(cdae):
                #print(f"{x_tmp.shape=}")
                x_tmp = layer(x_tmp)

            x_tmp = x_tmp.reshape(x_shape)

            # multiplicative skip connection
            if self.mask:
                x_tmp = x_tmp * mix

            ret[i, ...] = x_tmp

        return ret
