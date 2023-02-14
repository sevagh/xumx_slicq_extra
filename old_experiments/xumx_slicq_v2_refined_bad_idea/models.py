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
    phasemix_sep,
    overlap_add_slicq,
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
        self.umx = _UnmixAllTargets(jagged_slicq_sample_input, self.cnorm, max_bin, input_means, input_scales)

    def freeze(self):
        # set all parameters as not requiring gradient, more RAM-efficient
        # at test time
        for p in self.parameters():
            # p.requires_grad = False
            p.grad = None
        self.eval()

    def forward(self, x, return_nsgts: bool = False) -> Tensor:
        n_samples = x.shape[-1]

        X_complex = self.nsgt(x)
        Ycomplex_all = self.umx(X_complex)

        y_all = self.insgt(Ycomplex_all, n_samples)
        if return_nsgts:
            return y_all, Ycomplex_all
        return y_all


# inner class for doing umx for all targets for all slicqt blocks
class _UnmixAllTargets(nn.Module):
    def __init__(
        self,
        jagged_slicq_sample_input,
        cnorm,
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
                        cnorm,
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

    def forward(self, Xcomplex) -> Tensor:
        futures = [
            torch.jit.fork(self.sliced_umx[i], Xcomplex_block)
            for i, Xcomplex_block in enumerate(Xcomplex)
        ]
        Ycomplex = [torch.jit.wait(future) for future in futures]
        return Ycomplex



# inner class for doing umx for all targets per slicqt block
class _SlicedUnmix(nn.Module):
    def __init__(
        self,
        slicq_sample_input,
        cnorm,
        input_mean=None,
        input_scale=None,
        wiener_win_len: Optional[int] = 5000,
        softmask: bool = False,
        niter: int = 1,
    ):
        super(_SlicedUnmix, self).__init__()

        (
            nb_samples,
            nb_channels,
            nb_f_bins,
            nb_slices,
            nb_t_bins,
        ) = slicq_sample_input.shape

        self.cnorm = cnorm

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
        deoverlap = []

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
            if i == 1:
                decoder.append(Sigmoid())
            else:
                decoder.append(ReLU())

        cdae_1 = Sequential(*encoder, *decoder)
        cdae_2 = copy.deepcopy(cdae_1)
        cdae_3 = copy.deepcopy(cdae_1)
        cdae_4 = copy.deepcopy(cdae_1)

        self.cdaes = nn.ModuleList([cdae_1, cdae_2, cdae_3, cdae_4])
        self.mask = True

        # grow the overlap-added half dimension to its full size
        deoverlap_1 = ConvTranspose2d(nb_channels, nb_channels, (1, 3), stride=(1, 2), bias=True, dtype=torch.complex64)

        deoverlap_2 = copy.deepcopy(deoverlap_1)
        deoverlap_3 = copy.deepcopy(deoverlap_1)
        deoverlap_4 = copy.deepcopy(deoverlap_1)

        self.deoverlaps = nn.ModuleList([deoverlap_1, deoverlap_2, deoverlap_3, deoverlap_4])

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

        self.wiener_win_len = wiener_win_len
        self.niter = niter
        self.softmask = softmask

    def post_wiener(self, X, mag_slicqtgrams):
        # transposing it as
        # (nb_samples, nb_frames, nb_bins,{1,nb_channels}, nb_sources)
        mag_slicqtgrams = mag_slicqtgrams.permute(1, 4, 3, 2, 0)

        # rearranging it into:
        # (nb_samples, nb_frames, nb_bins, nb_channels) to feed
        # into filtering methods
        X = X.permute(0, 3, 2, 1)

        nb_frames = mag_slicqtgrams.shape[1]
        targets_slicqtgrams = torch.zeros(
            *X.shape + (4,),
            dtype=X.dtype,
            device=X.device,
        )

        pos = 0
        if self.wiener_win_len:
            wiener_win_len = self.wiener_win_len
        else:
            wiener_win_len = nb_frames
        while pos < nb_frames:
            cur_frame = torch.arange(pos, min(nb_frames, pos + wiener_win_len))
            pos = int(cur_frame[-1]) + 1

            targets_slicqtgrams[:, cur_frame, ...] = norbert.wiener(
                mag_slicqtgrams[:, cur_frame, ...],
                X[:, cur_frame, ...],
                self.niter,
                use_softmask=self.softmask,
            )

        # getting to (nb_samples, nb_targets, channel, fft_size, n_frames)
        targets_slicqtgrams = targets_slicqtgrams.permute(4, 0, 3, 2, 1).contiguous()
        return targets_slicqtgrams

    def freeze(self):
        # set all parameters as not requiring gradient, more RAM-efficient
        # at test time
        for p in self.parameters():
            # p.requires_grad = False
            p.grad = None
        self.eval()

    def forward(self, x_complex: Tensor) -> Tensor:
        # magnitude slicqtgram
        x = self.cnorm(x_complex)

        x_shape = x.shape
        nb_samples, nb_channels, nb_f_bins, nb_slices, nb_t_bins = x_shape

        # targets magnitude slicqtgram
        Ycomplex_all = torch.zeros((4, *x_complex.shape,), device=x.device, dtype=x_complex.dtype)

        x = overlap_add_slicq(x)
        mix = x.detach().clone()
        x_complex_ola = overlap_add_slicq(torch.view_as_complex(x_complex))
        Ymag_all_ola = torch.zeros((4, *x.shape,), device=x.device, dtype=x.dtype)

        # shift and scale input to mean=0 std=1 (across all bins)
        x = x.permute(0, 1, 3, 2)
        x += self.input_mean
        x *= self.input_scale
        x = x.permute(0, 1, 3, 2)

        for i, cdae in enumerate(self.cdaes):
            x_tmp = x.clone()
            for j, layer in enumerate(cdae):
                x_tmp = layer(x_tmp)

            # multiplicative skip connection
            if self.mask:
                x_tmp = x_tmp * mix

            Ymag_all_ola[i] = x_tmp

        Y_complex_ola = self.post_wiener(x_complex_ola, Ymag_all_ola)

        for i, deoverlap in enumerate(self.deoverlaps):
            y_tmp = deoverlap(Y_complex_ola[i])
            # crop
            y_tmp = y_tmp[:, :, : nb_f_bins, : nb_t_bins * nb_slices]

            y_tmp = y_tmp.reshape(x_shape)

            Ycomplex_all[i] = torch.view_as_real(y_tmp)

        return Ycomplex_all


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
        return torch.unsqueeze(x, dim=0).repeat(4, 1, 1, 1, 1, 1, 1)
