from typing import Optional
from tqdm import trange
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import (
    Parameter,
    Sequential,
    LSTM,
    Linear,
    BatchNorm1d,
    BatchNorm2d,
    Conv2d,
    ConvTranspose2d,
    Tanh,
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
        hidden_size_chans: int = 12,
        nb_layers: int = 3,
        unidirectional: bool = False,
        max_bin=None,
        input_means=None,
        input_scales=None,
    ):
        super(Unmix, self).__init__()

        self.hidden_size_chans = hidden_size_chans

        self.sliced_umx = nn.ModuleList()

        self.stop_idx = 0
        freq_idx = 0
        for i, C_block in enumerate(jagged_slicq_sample_input):
            input_mean = input_means[i] if input_means else None
            input_scale = input_scales[i] if input_scales else None

            freq_start = freq_idx

            if max_bin is not None and freq_start >= max_bin:
                self.stop_idx = i
                break
            else:
                self.sliced_umx.append(
                    _SlicedUnmix(
                        C_block,
                        hidden_size_chans,
                        input_mean=input_mean,
                        input_scale=input_scale,
                    )
                )

            # advance frequency pointer
            freq_idx += C_block.shape[2]

        hidden_size = total_freqs = self.stop_idx * hidden_size_chans
        print(f"lstm hidden size: {hidden_size}")

        if unidirectional:
            lstm_hidden_size = hidden_size
        else:
            lstm_hidden_size = hidden_size // 2

        lstm = LSTM(
            input_size=hidden_size,
            hidden_size=lstm_hidden_size,
            num_layers=nb_layers,
            bidirectional=not unidirectional,
            batch_first=False,
            dropout=0.4 if nb_layers > 1 else 0,
        )

        self.hidden_size = hidden_size

        lstm_1 = lstm
        lstm_2 = copy.deepcopy(lstm)
        lstm_3 = copy.deepcopy(lstm)
        lstm_4 = copy.deepcopy(lstm)

        fc_hidden_size = hidden_size * 2
        fc = Linear(in_features=fc_hidden_size, out_features=hidden_size, bias=False)
        bn = BatchNorm1d(hidden_size)

        fc_1 = fc
        fc_2 = copy.deepcopy(fc)
        fc_3 = copy.deepcopy(fc)
        fc_4 = copy.deepcopy(fc)

        bn_1 = bn
        bn_2 = copy.deepcopy(bn)
        bn_3 = copy.deepcopy(bn)
        bn_4 = copy.deepcopy(bn)

        self.lstms = nn.ModuleList([lstm_1, lstm_2, lstm_3, lstm_4])
        self.fcs = nn.ModuleList([fc_1, fc_2, fc_3, fc_4])
        self.bns = nn.ModuleList([bn_1, bn_2, bn_3, bn_4])

    def freeze(self):
        # set all parameters as not requiring gradient, more RAM-efficient
        # at test time
        for p in self.parameters():
            # p.requires_grad = False
            p.grad = None
        self.eval()

    def forward(self, Xmag_all) -> Tensor:
        # only operate on Xmags below bandwidth
        Xmag_above = [torch.unsqueeze(Xmag_above_bw, dim=0).repeat(4, 1, 1, 1, 1, 1) for Xmag_above_bw in Xmag_all[self.stop_idx:]]

        Xmag = Xmag_all[:self.stop_idx]
        nb_slices = Xmag[0].shape[-2]

        # store mixes to apply sigmoid (i.e. softmask) at final step
        mixes = [Xmag_block.clone() for Xmag_block in Xmag]

        downsampling_blocks = [
            torch.jit.fork(self.sliced_umx[i], Xmag_block)
            for i, Xmag_block in enumerate(Xmag)
        ]
        downsampled_repr = [torch.jit.wait(future) for future in downsampling_blocks]

        total_downsampled = torch.cat(downsampled_repr, dim=3)

        # 4 targets of umx
        for i in range(4):
            x_tmp = total_downsampled[i].clone()

            x_tmp = x_tmp.view(x_tmp.shape[0], -1, x_tmp.shape[-1])

            nb_samples, _, nb_frames = x_tmp.shape

            x_tmp = x_tmp.reshape(nb_frames, nb_samples, self.hidden_size)

            # apply 3-layers of stacked LSTM
            lstm_out = self.lstms[i](x_tmp)

            # lstm skip connection
            x_tmp = torch.cat([x_tmp, lstm_out[0]], -1)

            # first dense stage + batch norm + ReLU
            x_tmp = self.fcs[i](x_tmp.reshape(-1, x_tmp.shape[-1]))
            x_tmp = self.bns[i](x_tmp)
            x_tmp = F.relu(x_tmp)

            x_tmp = x_tmp.reshape(nb_samples, self.hidden_size_chans, -1, nb_frames)

            total_downsampled[i] = x_tmp

        to_upsample = torch.split(total_downsampled, 1, dim=3)

        upsampling_blocks = [
            torch.jit.fork(self.sliced_umx[i].upsample, Ymag_block, nb_slices)
            for i, Ymag_block in enumerate(to_upsample)
        ]

        upsampled_repr = [torch.jit.wait(future) for future in upsampling_blocks]

        # upsampled_repr are sigmoid soft masks, apply to mix
        ret = [upsampled_repr_ * mix[None, ...] for upsampled_repr_, mix in zip(upsampled_repr, mixes)]

        # append original blocks skipped that were above bandwidth, and return
        return ret+Xmag_above


# inner class for doing umx for all targets per slicqt block
class _SlicedUnmix(nn.Module):
    def __init__(
        self,
        slicq_sample_input,
        hidden_size: int,
        input_mean=None,
        input_scale=None,
    ):
        super(_SlicedUnmix, self).__init__()

        self.hidden_size = hidden_size

        (
            nb_samples,
            nb_channels,
            nb_f_bins,
            _,
            nb_t_bins,
        ) = slicq_sample_input.shape

        self.nb_f_bins = nb_f_bins
        self.nb_t_bins = nb_t_bins
        self.nb_channels = nb_channels

        window = nb_t_bins
        hop = window // 2

        # 1: first stage downsampling with Conv2d + Tanh to feed LSTM
        downsampler = [
            Conv2d(nb_channels, hidden_size, (nb_f_bins, window), stride=(1, hop), bias=False),
            BatchNorm2d(hidden_size),
            Tanh(),
        ]

        # LSTM happens here

        # 2. upsampler with Sigmoid for final soft mask
        upsampler = [
            ConvTranspose2d(hidden_size, nb_channels, (nb_f_bins, window), stride=(1, hop), bias=True),
            Sigmoid(),
        ]

        downsampler_1 = Sequential(*downsampler)
        downsampler_2 = copy.deepcopy(downsampler_1)
        downsampler_3 = copy.deepcopy(downsampler_1)
        downsampler_4 = copy.deepcopy(downsampler_1)

        upsampler_1 = Sequential(*upsampler)
        upsampler_2 = copy.deepcopy(upsampler_1)
        upsampler_3 = copy.deepcopy(upsampler_1)
        upsampler_4 = copy.deepcopy(upsampler_1)

        self.downsamplers = nn.ModuleList([downsampler_1, downsampler_2, downsampler_3, downsampler_4])
        self.upsamplers = nn.ModuleList([upsampler_1, upsampler_2, upsampler_3, upsampler_4])

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
        x_shape = x.shape
        (
            nb_samples,
            nb_channels,
            nb_f_bins,
            nb_slices,
            nb_t_bins,
        ) = x_shape

        # shift and scale input to mean=0 std=1 (across all bins)
        x = torch.flatten(x, start_dim=-2, end_dim=-1)
        x = x.permute(0, 1, 3, 2)
        x += self.input_mean
        x *= self.input_scale
        x = x.permute(0, 1, 3, 2)

        ret = [None]*4
        for i in range(4):
            x_tmp = x.clone()

            # apply downsampler conv+tanh
            for j, layer in enumerate(self.downsamplers[i]):
                x_tmp = layer(x_tmp)

            ret[i] = torch.unsqueeze(x_tmp, dim=0)

        return torch.cat(ret, dim=0)

    def upsample(self, x: Tensor, nb_slices: int) -> Tensor:
        nb_samples = x.shape[1]
        ret = [None]*4

        for i in range(4):
            # operate per-target
            x_tmp = x[i].clone()

            # apply upsampler conv+sigmoid
            for j, layer in enumerate(self.upsamplers[i]):
                x_tmp = layer(x_tmp)

            x_tmp = x_tmp.reshape(nb_samples, self.nb_channels, self.nb_f_bins, -1, self.nb_t_bins)
            ret[i] = torch.unsqueeze(x_tmp, dim=0)

        return torch.cat(ret, dim=0)
