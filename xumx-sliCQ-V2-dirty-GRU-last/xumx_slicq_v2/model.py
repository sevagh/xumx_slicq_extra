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
    Linear,
    BatchNorm1d,
    GRU,
    Sequential,
    Sigmoid,
    Tanh,
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
        pre_gru_hidden_size: int = 10,
        gru_hidden_size: int = 256,
        gru_layers: int = 3,
        gru_unidirectional: bool = False,
        hidden_size_1: int = 50,
        hidden_size_2: int = 50,
        input_means=None,
        input_scales=None,
    ):
        super(Unmix, self).__init__()

        self.sliced_umx = nn.ModuleList()
        self.gru_hidden_size = gru_hidden_size

        freq_idx = 0
        for i, C_block in enumerate(jagged_slicq_sample_input):
            input_mean = input_means[i] if input_means else None
            input_scale = input_scales[i] if input_scales else None

            freq_start = freq_idx

            self.sliced_umx.append(
                _SlicedUnmix(
                    C_block,
                    hidden_size_1=hidden_size_1,
                    input_mean=input_mean,
                    input_scale=input_scale,
                )
            )

            # advance frequency pointer
            freq_idx += C_block.shape[2]

        pre_gru = []

        # pre-GRU bottleneck layer

        # first, 1x1 conv to reduce channels
        pre_gru = [
            Conv2d(
                hidden_size_2,
                pre_gru_hidden_size,
                (1, 1),
                bias=False,
            ),
            BatchNorm2d(pre_gru_hidden_size),
            ReLU(),
        ]

        # GRU for recurrent part of network
        gru = [
            Linear(
                in_features=255*pre_gru_hidden_size,
                out_features=gru_hidden_size,
                bias=False
            ),
            # tanh activation before GRU
            BatchNorm1d(gru_hidden_size),
            Tanh(),
            GRU(
                input_size=gru_hidden_size,
                hidden_size=gru_hidden_size if gru_unidirectional else gru_hidden_size//2,
                num_layers=gru_layers,
                bidirectional=not gru_unidirectional,
                dropout=0.4 if gru_layers > 1 else 0,
                batch_first=False,
            ),
            Linear(
                in_features=gru_hidden_size*2,
                out_features=255*pre_gru_hidden_size,
                bias=False,
            ),
            BatchNorm1d(255*pre_gru_hidden_size),
            ReLU(),
        ]

        # post-GRU 1x1 to increase channels, end of bottleneck
        post_gru = [
            Conv2d(
                pre_gru_hidden_size,
                hidden_size_2,
                (1, 1),
                bias=False,
            ),
            BatchNorm2d(hidden_size_2),
            ReLU(),
        ]

        pre_gru_1 = Sequential(*pre_gru)
        pre_gru_2 = copy.deepcopy(pre_gru_1)
        pre_gru_3 = copy.deepcopy(pre_gru_1)
        pre_gru_4 = copy.deepcopy(pre_gru_1)

        gru_1 = Sequential(*gru)
        gru_2 = copy.deepcopy(gru_1)
        gru_3 = copy.deepcopy(gru_1)
        gru_4 = copy.deepcopy(gru_1)

        post_gru_1 = Sequential(*post_gru)
        post_gru_2 = copy.deepcopy(post_gru_1)
        post_gru_3 = copy.deepcopy(post_gru_1)
        post_gru_4 = copy.deepcopy(post_gru_1)

        self.pre_grus = nn.ModuleList([pre_gru_1, pre_gru_2, pre_gru_3, pre_gru_4])
        self.grus = nn.ModuleList([gru_1, gru_2, gru_3, gru_4])
        self.post_grus = nn.ModuleList([post_gru_1, post_gru_2, post_gru_3, post_gru_4])

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

        bottlenecked_global_encoded = [None]*4
        for i in range(4):
            x_tmp = global_encoded[i]
            #print(f"x_tmp: {x_tmp.shape}")

            # apply pre-gru compression/bottleneck
            x_tmp = self.pre_grus[i](x_tmp)
            #print(f"x_tmp: {x_tmp.shape}")

            nb_samples, nb_channels, nb_bins, nb_frames = x_tmp.shape

            x_tmp = x_tmp.reshape(-1, nb_channels * nb_bins)
            #print(f"x_tmp: {x_tmp.shape}")

            # first fc layer + tanh activation
            x_tmp = self.grus[i][0](x_tmp)
            x_tmp = self.grus[i][1](x_tmp)
            x_tmp = self.grus[i][2](x_tmp)

            # pre-GRU
            x_tmp = x_tmp.reshape(nb_frames, nb_samples, self.gru_hidden_size)

            # apply GRU
            gru_out = self.grus[i][3](x_tmp)

            # skip connection
            x_tmp = torch.cat([x_tmp, gru_out[0]], -1)

            x_tmp = x_tmp.reshape(-1, x_tmp.shape[-1])

            # post fc layer
            x_tmp = self.grus[i][4](x_tmp)
            x_tmp = self.grus[i][5](x_tmp)
            x_tmp = self.grus[i][6](x_tmp)

            # back to decoder part of cdae
            x_tmp = x_tmp.reshape(nb_samples, nb_channels, nb_bins, nb_frames)

            # grow back to original channels
            x_tmp = self.post_grus[i](x_tmp)

            bottlenecked_global_encoded[i] = torch.unsqueeze(x_tmp, dim=0)

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
        hidden_size_1: int,
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

        encoder = [
            Conv2d(
                nb_channels,
                hidden_size_1,
                (freq_filter, window),
                stride=(1, hop),
                bias=False,
            ),
            BatchNorm2d(hidden_size_1),
            ReLU(),
        ]

        decoder = [
            ConvTranspose2d(
                hidden_size_1,
                nb_channels,
                (freq_filter, window),
                stride=(1, hop),
                bias=True,
            ),
            Sigmoid(),
        ]

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

            ret[i] = torch.unsqueeze(x_tmp, dim=0)

        return torch.cat(ret, dim=0)

    def decode(self, x: Tensor) -> Tensor:
        ret = [None]*4

        for i, cdae in enumerate(self.cdaes):
            # apply last 5 layers i.e. decoder (convT->batchnormn->relu, convT->sigmoid)
            x_tmp = x[i].clone()

            x_tmp = cdae[3](x_tmp)
            x_tmp = cdae[4](x_tmp)

            ret[i] = torch.unsqueeze(x_tmp, dim=0)

        return torch.cat(ret, dim=0)

    def forward(self, x: Tensor) -> Tensor:
        x = self.encode(x)
        x = self.decode(x)
        return x
