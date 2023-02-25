from typing import Optional
from tqdm import trange
import torch
import torch.nn as nn
import norbert
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
from .phase import blockwise_wiener, blockwise_phasemix_sep
import copy


# inner class for doing umx for all targets for all slicqt blocks
class Unmix(nn.Module):
    def __init__(
        self,
        jagged_slicq_sample_input,
        hidden_size_1: int = 50,
        hidden_size_2: int = 51,
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

    def freeze(self):
        # set all parameters as not requiring gradient, more RAM-efficient
        # at test time
        for p in self.parameters():
            # p.requires_grad = False
            p.grad = None
        self.eval()

    def forward(self, Xcomplex, return_masks=False, wiener=True) -> Tensor:
        Ycomplex = [None]*len(Xcomplex)
        Ymasks = [None]*len(Xcomplex)

        for i, Xblock in enumerate(Xcomplex):
            Ycomplex_block, Ymask_block = self.sliced_umx[i](
                Xblock, torch.abs(torch.view_as_complex(Xblock)), wiener=wiener
            )
            Ycomplex[i] = Ycomplex_block
            Ymasks[i] = Ymask_block

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
            ReLU(),
        ])

        encoder.extend([
            Conv2d(
                hidden_size_1,
                hidden_size_2,
                (freq_filter, time_filter_2),
                bias=False,
            ),
            BatchNorm2d(hidden_size_2),
            ReLU(),
        ])

        decoder.extend([
            ConvTranspose2d(
                hidden_size_2,
                hidden_size_1,
                (freq_filter, time_filter_2),
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

    def forward(self, xcomplex: Tensor, x: Tensor, wiener: bool = True) -> Tensor:
        mix = x.detach().clone()

        x_shape = x.shape
        nb_samples, nb_channels, nb_f_bins, nb_slices, nb_t_bins = x_shape

        ret = torch.zeros((4, *x_shape,), device=x.device, dtype=x.dtype)
        ret_masks = torch.zeros((4, *x_shape,), device=x.device, dtype=x.dtype)

        x = torch.flatten(x, start_dim=-2, end_dim=-1)

        # shift and scale input to mean=0 std=1 (across all bins)
        x = x.permute(0, 1, 3, 2)
        x += self.input_mean
        x *= self.input_scale
        x = x.permute(0, 1, 3, 2)

        x_tmp_1 = x.clone()
        x_tmp_2 = x.clone()
        x_tmp_3 = x.clone()
        x_tmp_4 = x.clone()

        # 4 targets, encoder 1
        x_tmp_1 = self.cdaes[0][0](x_tmp_1)
        x_tmp_1 = self.cdaes[0][1](x_tmp_1)
        x_tmp_1 = self.cdaes[0][2](x_tmp_1)

        x_tmp_2 = self.cdaes[1][0](x_tmp_2)
        x_tmp_2 = self.cdaes[1][1](x_tmp_2)
        x_tmp_2 = self.cdaes[1][2](x_tmp_2)

        x_tmp_3 = self.cdaes[2][0](x_tmp_3)
        x_tmp_3 = self.cdaes[2][1](x_tmp_3)
        x_tmp_3 = self.cdaes[2][2](x_tmp_3)

        x_tmp_4 = self.cdaes[3][0](x_tmp_4)
        x_tmp_4 = self.cdaes[3][1](x_tmp_4)
        x_tmp_4 = self.cdaes[3][2](x_tmp_4)

        # cross-target skip connection
        x_skip = (x_tmp_1+x_tmp_2+x_tmp_3+x_tmp_4).clone()

        # encoder 2, decoder 1
        x_tmp_1 = self.cdaes[0][3](x_tmp_1)
        x_tmp_1 = self.cdaes[0][4](x_tmp_1)
        x_tmp_1 = self.cdaes[0][5](x_tmp_1)
        x_tmp_1 = self.cdaes[0][6](x_tmp_1)
        x_tmp_1 = self.cdaes[0][7](x_tmp_1)
        x_tmp_1 = self.cdaes[0][8](x_tmp_1)

        x_tmp_2 = self.cdaes[1][3](x_tmp_2)
        x_tmp_2 = self.cdaes[1][4](x_tmp_2)
        x_tmp_2 = self.cdaes[1][5](x_tmp_2)
        x_tmp_2 = self.cdaes[1][6](x_tmp_2)
        x_tmp_2 = self.cdaes[1][7](x_tmp_2)
        x_tmp_2 = self.cdaes[1][8](x_tmp_2)

        x_tmp_3 = self.cdaes[2][3](x_tmp_3)
        x_tmp_3 = self.cdaes[2][4](x_tmp_3)
        x_tmp_3 = self.cdaes[2][5](x_tmp_3)
        x_tmp_3 = self.cdaes[2][6](x_tmp_3)
        x_tmp_3 = self.cdaes[2][7](x_tmp_3)
        x_tmp_3 = self.cdaes[2][8](x_tmp_3)

        x_tmp_4 = self.cdaes[3][3](x_tmp_4)
        x_tmp_4 = self.cdaes[3][4](x_tmp_4)
        x_tmp_4 = self.cdaes[3][5](x_tmp_4)
        x_tmp_4 = self.cdaes[3][6](x_tmp_4)
        x_tmp_4 = self.cdaes[3][7](x_tmp_4)
        x_tmp_4 = self.cdaes[3][8](x_tmp_4)

        # cross-target skip conn before decoder 2
        x_tmp_1 = x_tmp_1 + x_skip
        x_tmp_2 = x_tmp_2 + x_skip
        x_tmp_3 = x_tmp_3 + x_skip
        x_tmp_4 = x_tmp_4 + x_skip

        # decoder 2 + mask (i.e. multiplicative skip conn)
        x_tmp_1 = self.cdaes[0][9](x_tmp_1)
        x_tmp_1 = self.cdaes[0][10](x_tmp_1)

        x_tmp_2 = self.cdaes[1][9](x_tmp_2)
        x_tmp_2 = self.cdaes[1][10](x_tmp_2)

        x_tmp_3 = self.cdaes[2][9](x_tmp_3)
        x_tmp_3 = self.cdaes[2][10](x_tmp_3)

        x_tmp_4 = self.cdaes[3][9](x_tmp_4)
        x_tmp_4 = self.cdaes[3][10](x_tmp_4)

        x_tmp_1 = x_tmp_1.reshape(x_shape)
        x_tmp_2 = x_tmp_2.reshape(x_shape)
        x_tmp_3 = x_tmp_3.reshape(x_shape)
        x_tmp_4 = x_tmp_4.reshape(x_shape)

        # store the sigmoid/soft mask before multiplying with mix
        ret_masks[0] = x_tmp_1.clone()
        ret_masks[1] = x_tmp_2.clone()
        ret_masks[2] = x_tmp_3.clone()
        ret_masks[3] = x_tmp_4.clone()

        # multiplicative skip connection i.e. soft mask
        ret[0] = x_tmp_1 * mix
        ret[1] = x_tmp_2 * mix
        ret[2] = x_tmp_3 * mix
        ret[3] = x_tmp_4 * mix

        # embedded blockwise wiener-EM (flattened in function then unflattened)
        if wiener:
            ret = blockwise_wiener(xcomplex, ret)
        else:
            ret = blockwise_phasemix_sep(xcomplex, ret)

        # also return the mask
        return ret, ret_masks
