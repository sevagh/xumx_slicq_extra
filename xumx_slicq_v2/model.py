from typing import Optional
from tqdm import trange
import torch
import torch.nn as nn
import norbert
import torch.nn.functional as F
from torch import Tensor
from torch.nn import (
    Parameter,
    LeakyReLU,
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
from .phase import blockwise_wiener
import copy


# inner class for doing umx for all targets for all slicqt blocks
class Unmix(nn.Module):
    def __init__(
        self,
        jagged_slicq_sample_input,
        hidden_size_1: int = 50,
        hidden_size_2: int = 51,
        bottleneck_hidden_size: int = 13,
        bottleneck_freq_filter: int = 7,
        bottleneck_time_filter: int = 3,
        freq_filter_small: int = 1,
        freq_filter_medium: int = 3,
        freq_filter_large: int = 5,
        freq_thresh_small: int = 10,
        freq_thresh_medium: int = 20,
        time_filter_2: int = 4,
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

            self.sliced_umx.append(
                _SlicedUnmix(
                    C_block,
                    hidden_size_1=hidden_size_1,
                    hidden_size_2=hidden_size_2,
                    freq_filter_small=freq_filter_small,
                    freq_thresh_small=freq_thresh_small,
                    freq_filter_medium=freq_filter_medium,
                    freq_thresh_medium=freq_thresh_medium,
                    freq_filter_large=freq_filter_large,
                    time_filter_2=time_filter_2,
                    input_mean=input_mean,
                    input_scale=input_scale,
                )
            )

            # advance frequency pointer
            freq_idx += C_block.shape[2]

        bottleneck = []

        # ResNet-like bottleneck layer

        # first, 1x1 conv to reduce channels
        bottleneck.extend([
            Conv2d(
                hidden_size_2,
                bottleneck_hidden_size,
                (1, 1),
                bias=False,
            ),
            BatchNorm2d(bottleneck_hidden_size),
            LeakyReLU(),
        ])

        # next: actual bottleneck layer
        bottleneck.extend([
            Conv2d(
                bottleneck_hidden_size,
                bottleneck_hidden_size,
                (bottleneck_freq_filter, bottleneck_time_filter),
                bias=False,
            ),
            BatchNorm2d(bottleneck_hidden_size),
            LeakyReLU(),
        ])

        # finally, 1x1 to increase channels, end of bottleneck
        bottleneck.extend([
            Conv2d(
                bottleneck_hidden_size,
                hidden_size_2,
                (1, 1),
                bias=False,
            ),
            BatchNorm2d(hidden_size_2),
            LeakyReLU(),
        ])

        bottleneck_1 = Sequential(*bottleneck)
        bottleneck_2 = copy.deepcopy(bottleneck_1)
        bottleneck_3 = copy.deepcopy(bottleneck_1)
        bottleneck_4 = copy.deepcopy(bottleneck_1)

        self.bottlenecks = nn.ModuleList([bottleneck_1, bottleneck_2, bottleneck_3, bottleneck_4])

        self.bottleneck_freq_pad = bottleneck_freq_filter-1
        self.bottleneck_time_pad = bottleneck_time_filter-1

    def freeze(self):
        # set all parameters as not requiring gradient, more RAM-efficient
        # at test time
        for p in self.parameters():
            # p.requires_grad = False
            p.grad = None
        self.eval()

    def forward(self, Xcomplex, return_masks=False) -> Tensor:
        n_blocks = len(Xcomplex)

        # store mixed magnitude slicqt
        mixes = [torch.abs(torch.view_as_complex(Xblock)) for Xblock in Xcomplex]

        encoded = [None]*n_blocks
        Ycomplex = [None]*n_blocks
        Ymasks = [None]*n_blocks

        # flatten and apply encoder per-block
        for i, mix in enumerate(mixes):
            x = mix.clone()
            x = torch.flatten(x, start_dim=-2, end_dim=-1)
            x = self.sliced_umx[i].encode(x)
            encoded[i] = x

        # concatenate by frequency dimension, store frequency bins for deconcatenation later
        deconcat_indexes = [encoded_elem.shape[-2] for encoded_elem in encoded]
        global_encoded = torch.cat(encoded, dim=-2)

        # apply bottleneck per-target
        # pad before bottleneck
        global_encoded = F.pad(global_encoded, (0, self.bottleneck_time_pad, 0, self.bottleneck_freq_pad), "constant", 0)

        bottlenecked_global_encoded = [None]*4
        for i in range(4):
            bottlenecked_global_encoded[i] = torch.unsqueeze(self.bottlenecks[i](global_encoded[i]), dim=0)

        global_encoded = torch.cat(bottlenecked_global_encoded, dim=0)

        # deconcatenate after global bottleneck layer
        frequency_ptr = 0
        for i, deconcat_index in enumerate(deconcat_indexes):
            encoded[i] = global_encoded[..., frequency_ptr : frequency_ptr + deconcat_index, :]
            frequency_ptr += deconcat_index

        # apply decoder per-block
        for i, mix in enumerate(mixes):
            decoded_mask = self.sliced_umx[i].decode(encoded[i])

            # unflatten by reshaping
            Ymasks[i] = decoded_mask.reshape((4, *mix.shape,))

            # multiplicative skip connection i.e. soft mask per-block
            masked_slicqt = mix*Ymasks[i]

            # blockwise wiener-EM
            Ycomplex[i] = blockwise_wiener(Xcomplex[i], masked_slicqt)

        if return_masks:
            return Ycomplex, Ymasks
        return Ycomplex


# inner class for doing umx for all targets per slicqt block
class _SlicedUnmix(nn.Module):
    def __init__(
        self,
        slicq_sample_input,
        hidden_size_1: int = 25,
        hidden_size_2: int = 55,
        freq_filter_small: int = 1,
        freq_thresh_small: int = 10,
        freq_filter_medium: int = 3,
        freq_thresh_medium: int = 20,
        freq_filter_large: int = 5,
        time_filter_2: int = 3,
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

        if nb_f_bins < freq_thresh_small:
            freq_filter = freq_filter_small
        elif nb_f_bins < freq_thresh_medium:
            freq_filter = freq_filter_medium
        else:
            freq_filter = freq_filter_large

        encoder = []
        decoder = []

        window = nb_t_bins
        hop = window // 2

        encoder.extend([
            Conv2d(
                nb_channels,
                hidden_size_1,
                (freq_filter, window),
                stride=(1, hop),
                bias=False,
            ),
            BatchNorm2d(hidden_size_1),
            LeakyReLU(),
        ])

        encoder.extend([
            Conv2d(
                hidden_size_1,
                hidden_size_2,
                (freq_filter, time_filter_2),
                bias=False,
            ),
            BatchNorm2d(hidden_size_2),
            LeakyReLU(),
        ])

        decoder.extend([
            ConvTranspose2d(
                hidden_size_2,
                hidden_size_1,
                (freq_filter, time_filter_2),
                bias=False,
            ),
            BatchNorm2d(hidden_size_1),
            LeakyReLU(),
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

    def encode(self, x: Tensor) -> Tensor:
        ret = [None]*4

        # shift and scale input to mean=0 std=1 (across all bins)
        x = x.permute(0, 1, 3, 2)
        x += self.input_mean
        x *= self.input_scale
        x = x.permute(0, 1, 3, 2)

        for i, cdae in enumerate(self.cdaes):
            # apply first 6 layers i.e. encoder (conv->batchnorm->relu x 2)
            x_tmp = x.clone()
            x_tmp = cdae[0](x_tmp)
            x_tmp = cdae[1](x_tmp)
            x_tmp = cdae[2](x_tmp)
            x_tmp = cdae[3](x_tmp)
            x_tmp = cdae[4](x_tmp)
            x_tmp = cdae[5](x_tmp)

            ret[i] = torch.unsqueeze(x_tmp, dim=0)

        return torch.cat(ret, dim=0)

    def decode(self, x: Tensor) -> Tensor:
        ret = [None]*4

        for i, cdae in enumerate(self.cdaes):
            # apply last 5 layers i.e. decoder (convT->batchnormn->relu, convT->sigmoid)
            x_tmp = x[i].clone()

            x_tmp = cdae[6](x_tmp)
            x_tmp = cdae[7](x_tmp)
            x_tmp = cdae[8](x_tmp)
            x_tmp = cdae[9](x_tmp)
            x_tmp = cdae[10](x_tmp)

            ret[i] = torch.unsqueeze(x_tmp, dim=0)

        return torch.cat(ret, dim=0)

    def forward(self, x: Tensor) -> Tensor:
        x = self.encode(x)
        x = self.decode(x)
        return x
