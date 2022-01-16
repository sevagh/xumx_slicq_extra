from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter, ReLU, BatchNorm2d, BatchNorm3d, Conv2d, ConvTranspose2d, Sequential, ModuleList, Linear
from .filtering import atan2, wiener
from .transforms import make_filterbanks_slicqt, ComplexNormSliCQT, phasemix_sep, NSGTBase
from collections import defaultdict
import numpy as np
import copy
import math

eps = 1.e-10


class DeOverlapNet(nn.Module):
    def __init__(
        self,
        nsgt,
        slicq_sample_input,
        input_mean=None,
        input_scale=None,
    ):
        super(DeOverlapNet, self).__init__()

        self.nsgt = nsgt

        interp = self.nsgt.interpolate(slicq_sample_input)
        interp_ola = self.nsgt.overlap_add(interp)

        self.nb_f_bins = interp_ola.shape[-2]
        self.nb_channels = interp_ola.shape[1]
        self.max_t_bins = interp.shape[-1]

        self.nb_m_bins = self.max_t_bins
        self.nwin = self.nb_m_bins
        self.hop = self.nwin//2

        # magic learnable window for the deoverlap
        self.deoverlap_window = Sequential(
            Linear(in_features=2*self.nwin, out_features=self.nwin, bias=False),
            BatchNorm2d(self.nb_channels),
            ReLU()
        )

        # stack of 1x1 convoluation layers to refine the deinterpolation
        deinterp_layers = []
        for i, slicq_ in enumerate(slicq_sample_input):
            nb_m_bins = slicq_.shape[-1]
            deinterp_layers.append(
                Sequential(
                   Linear(in_features=self.max_t_bins, out_features=nb_m_bins, bias=False),
                   BatchNorm3d(self.nb_channels),
                   ReLU()
                )
            )

        self.deinterp_layers = ModuleList(deinterp_layers)

        if input_mean is not None:
            input_mean = torch.from_numpy(-input_mean).float()
        else:
            input_mean = torch.zeros(self.nb_f_bins)

        if input_scale is not None:
            input_scale = torch.from_numpy(1.0 / input_scale).float()
        else:
            input_scale = torch.ones(self.nb_f_bins)

        self.input_mean = Parameter(input_mean)
        self.input_scale = Parameter(input_scale)

    def freeze(self):
        for p in self.parameters():
            p.grad = None
        self.eval()

    def forward(self, x: Tensor, nb_slices, ragged_shapes) -> Tensor:
        # apply inference in chunks of ncoefs, to have a fixed-size spectrogram to reason about

        # shift and scale input to mean=0 std=1 (across all bins)
        x = x.permute(0, 1, 3, 2)
        x += self.input_mean
        x *= self.input_scale
        x = x.permute(0, 1, 3, 2)

        # input is the overlap-added, interpolated sliCQT spectrogram
        ncoefs = x.shape[-1]
        nb_samples = x.shape[0]

        # deoverlap with linear layer
        x = self.deoverlap_add(x, nb_slices)
        x = self.deinterpolate(x, ragged_shapes)

        return x

    def deoverlap_add(self, slicq, nb_slices):
        nb_samples, nb_channels, nb_f_bins, ncoefs = slicq.shape

        out = torch.zeros((nb_samples, nb_channels, nb_f_bins, nb_slices, self.nb_m_bins), dtype=slicq.dtype, device=slicq.device)

        # each slice considers a hop of to the left and a hop to the right, in addition to the nwin of the middle

        # first slice is a special case pad left by hop zeros
        out[:, :, :, 0, :] = self.deoverlap_window(F.pad(slicq[:, :, :, : self.nwin+self.hop], (self.hop, 0)))

        ptr = self.hop
        for i in range(1, nb_slices):
            left_idx = ptr-self.hop
            right_idx = ptr+self.nwin+self.hop

            tmp1 = slicq[:, :, :, left_idx:right_idx]

            if tmp1.shape[-1] < 2*self.nwin:
                tmp1 = F.pad(tmp1, (0, self.hop))

            tmp2 = self.deoverlap_window(tmp1)

            # inverse of overlap-add
            out[:, :, :, i, :] = tmp2
            ptr += self.hop

        return out

    def deinterpolate(self, interpolated_slicq, ragged_shapes):
        max_time = interpolated_slicq.shape[-1]
        full_slicq = []
        fbin_ptr = 0
        for i, bucket_shape in enumerate(ragged_shapes):
            curr_slicq = torch.zeros(bucket_shape, dtype=interpolated_slicq.dtype, device=interpolated_slicq.device)

            small_time = bucket_shape[-1]
            slices = bucket_shape[-2]
            freqs = bucket_shape[-3]

            curr_slicq = self.deinterp_layers[i](interpolated_slicq[:, :, fbin_ptr:fbin_ptr+freqs, :])

            full_slicq.append(curr_slicq)

            fbin_ptr += freqs
        return full_slicq


class OpenUnmixCore(nn.Module):
    def __init__(
        self,
        slicq_sample_input,
        input_mean=None,
        input_scale=None,
    ):
        super(OpenUnmixCore, self).__init__()

        nb_channels = slicq_sample_input.shape[1]
        nb_bins = slicq_sample_input.shape[-2]
        self.nb_bins = nb_bins

        channels = [nb_channels, 25, 55, 75]
        layers = len(channels)-1

        filters = [(15, 47), (9, 21), (5, 13)]
        dilations = [(1, 2), (1, 2), (1, 2)]
        strides = [(1, 2), (1, 2), (1, 2)]

        encoder = []
        decoder = []

        for i in range(layers):
            encoder.append(
                Conv2d(
                    channels[i],
                    channels[i+1],
                    filters[i],
                    dilation=dilations[i],
                    stride=strides[i],
                    bias=False
                )
            )
            encoder.append(
                BatchNorm2d(channels[i+1]),
            )
            encoder.append(
                ReLU(),
            )

        for i in range(layers,0,-1):
            decoder.append(
                ConvTranspose2d(channels[i],
                    channels[i-1],
                    filters[i-1],
                    dilation=dilations[i-1],
                    stride=strides[i-1],
                    output_padding=(0, 1),
                    bias=False
                )
            )
            decoder.append(
                BatchNorm2d(channels[i-1])
            )
            decoder.append(
                ReLU()
            )

        self.cdae = Sequential(*encoder, *decoder)

        if input_mean is not None:
            input_mean = torch.from_numpy(-input_mean).float()
        else:
            input_mean = torch.zeros(self.nb_bins)

        if input_scale is not None:
            input_scale = torch.from_numpy(1.0 / input_scale).float()
        else:
            input_scale = torch.ones(self.nb_bins)

        self.input_mean = Parameter(input_mean)
        self.input_scale = Parameter(input_scale)

    def freeze(self):
        for p in self.parameters():
            p.grad = None
        self.eval()

    def forward(self, x: Tensor) -> Tensor:
        mix = x.detach().clone()

        # shift and scale input to mean=0 std=1 (across all bins)
        x = x.permute(0, 1, 3, 2)
        x = x + self.input_mean
        x = x * self.input_scale
        x = x.permute(0, 1, 3, 2)

        # apply CDAE here
        for i, layer in enumerate(self.cdae):
            x = layer(x)

        # crop
        x = x[..., : mix.shape[-1]]

        return x * mix


class OpenUnmix(nn.Module):
    """OpenUnmix Core spectrogram based separation module.

    Args:
        nb_bins, M (int): Number of sliCQ-NSGT tf bins
        nb_channels (int): Number of input audio channels (Default: `2`).
    """
    def __init__(
        self,
        nsgt,
        jagged_slicq_sample_input,
        model_b=False,
        input_mean=None,
        input_scale=None,
    ):
        super(OpenUnmix, self).__init__()

        self.unmix_vocals = OpenUnmixCore(
            nsgt.overlap_add(nsgt.interpolate(jagged_slicq_sample_input)),
            input_mean=input_mean,
            input_scale=input_scale,
        )

        self.unmix_other = copy.deepcopy(self.unmix_vocals)
        self.unmix_bass = copy.deepcopy(self.unmix_vocals)
        self.unmix_drums = copy.deepcopy(self.unmix_vocals)

        self.deoverlapnet = DeOverlapNet(
            nsgt,
            jagged_slicq_sample_input,
            input_mean=input_mean,
            input_scale=input_scale,
        )

    def freeze(self):
        for p in self.parameters():
            p.grad = None
        self.eval()

    def forward(self, X, nb_slices, ragged_shapes) -> Tensor:
        Y_vocals_interp_ola = self.unmix_vocals(X.detach().clone())
        Y_bass_interp_ola = self.unmix_bass(X.detach().clone())
        Y_other_interp_ola = self.unmix_other(X.detach().clone())
        Y_drums_interp_ola = self.unmix_drums(X.detach().clone())

        Y_vocals = self.deoverlapnet(Y_vocals_interp_ola, nb_slices, ragged_shapes)
        Y_bass = self.deoverlapnet(Y_bass_interp_ola, nb_slices, ragged_shapes)
        Y_other = self.deoverlapnet(Y_other_interp_ola, nb_slices, ragged_shapes)
        Y_drums = self.deoverlapnet(Y_drums_interp_ola, nb_slices, ragged_shapes)

        return Y_bass, Y_vocals, Y_other, Y_drums


class Separator(nn.Module):
    def __init__(
        self,
        xumx_model,
        xumx_nsgt,
        sample_rate: float = 44100.0,
        nb_channels: int = 2,
        device: str = "cpu",
        niter: int = 1,
        softmask: bool = False,
        wiener_win_len: Optional[int] = 300,
        n_fft: Optional[int] = 4096,
        n_hop: Optional[int] = 1024,
    ):
        super(Separator, self).__init__()
        # saving parameters
        self.niter = niter
        return Y_bass, Y_vocals, Y_other, Y_drums


class Separator(nn.Module):
    def __init__(
        self,
        xumx_model,
        xumx_nsgt,
        sample_rate: float = 44100.0,
        nb_channels: int = 2,
        device: str = "cpu",
        niter: int = 1,
        max_samples: int = 2_000_000,
        softmask: bool = False,
        wiener_win_len: Optional[int] = 300,
        n_fft: Optional[int] = 4096,
        n_hop: Optional[int] = 1024,
    ):
        super(Separator, self).__init__()
        # saving parameters
        self.niter = niter
        self.softmask = softmask
        self.max_samples = max_samples

        self.device = device

        self.nsgt, self.insgt = make_filterbanks_slicqt(
            xumx_nsgt, sample_rate=sample_rate
        )

        self.complexnorm = ComplexNormSliCQT(mono=nb_channels == 1)
        self.nb_channels = nb_channels

        self.xumx_model = xumx_model
        self.register_buffer("sample_rate", torch.as_tensor(sample_rate))

        self.n_fft = n_fft
        self.n_hop = n_hop
        self.wiener_win_len = wiener_win_len

        self.ordered_targets = ["vocals", "drums", "bass", "other"]

    def freeze(self):
        # set all parameters as not requiring gradient, more RAM-efficient
        # at test time
        for p in self.parameters():
            #p.requires_grad = False
            p.grad = None
        self.xumx_model.freeze()
        self.eval()

    @torch.no_grad()
    def forward(self, audio: Tensor) -> Tensor:
        """Performing the separation on audio input

        Args:
            audio (Tensor): [shape=(nb_samples, nb_channels, nb_timesteps)]
                mixture audio waveform

        Returns:
            Tensor: stacked tensor of separated waveforms
                shape `(nb_samples, nb_targets, nb_channels, nb_timesteps)`
        """
        nb_sources = 4
        nb_samples = audio.shape[0]

        print('Computing sliCQTs...')

        # cumulative waveforms
        y_bass_cum = []
        y_vocals_cum = []
        y_drums_cum = []
        y_other_cum = []

        for audio_short in torch.split(audio, self.max_samples, dim=-1):
            X = self.nsgt(audio_short)
            Xmag = self.complexnorm(X)
            ragged_shapes = [X_.shape for X_ in Xmag]
            Xmag_interp = self.nsgt.interpolate(Xmag)
            nb_slices = Xmag_interp.shape[-2]
            Xmag_interp_ola = self.nsgt.overlap_add(Xmag_interp)

            print('xumx-sliCQ inference...')

            # xumx inference - ragged magnitude slicq estimate
            Ymag_bass, Ymag_vocals, Ymag_other, Ymag_drums = self.xumx_model(Xmag_interp_ola, nb_slices, ragged_shapes)

            # initial mix phase + magnitude estimate
            Ycomplex_bass = phasemix_sep(X, Ymag_bass)
            Ycomplex_vocals = phasemix_sep(X, Ymag_vocals)
            Ycomplex_drums = phasemix_sep(X, Ymag_drums)
            Ycomplex_other = phasemix_sep(X, Ymag_other)

            y_bass_cum.append(self.insgt(Ycomplex_bass, audio_short.shape[-1]))
            y_drums_cum.append(self.insgt(Ycomplex_drums, audio_short.shape[-1]))
            y_other_cum.append(self.insgt(Ycomplex_other, audio_short.shape[-1]))
            y_vocals_cum.append(self.insgt(Ycomplex_vocals, audio_short.shape[-1]))

        # concat the cumulative stuff

        y_bass = torch.cat(y_bass_cum, dim=-1)[..., : audio.shape[-1]]
        y_drums = torch.cat(y_drums_cum, dim=-1)[..., : audio.shape[-1]]
        y_other = torch.cat(y_other_cum, dim=-1)[..., : audio.shape[-1]]
        y_vocals = torch.cat(y_vocals_cum, dim=-1)[..., : audio.shape[-1]]

        print('STFT Wiener-EM')

        # initial estimate was obtained with slicq
        # now we switch to the STFT domain for the wiener step

        audio = torch.squeeze(audio, dim=0)
        
        mix_stft = torch.view_as_real(torch.stft(audio, self.n_fft, hop_length=self.n_hop, return_complex=True))
        X = torch.abs(torch.view_as_complex(mix_stft))
        
        # initializing spectrograms variable
        spectrograms = torch.zeros(X.shape + (nb_sources,), dtype=audio.dtype, device=X.device)

        for j, target_name in enumerate(self.ordered_targets):
            # apply current model to get the source spectrogram
            if target_name == 'bass':
                target_est = torch.squeeze(y_bass, dim=0)
            elif target_name == 'vocals':
                target_est = torch.squeeze(y_vocals, dim=0)
            elif target_name == 'drums':
                target_est = torch.squeeze(y_drums, dim=0)
            elif target_name == 'other':
                target_est = torch.squeeze(y_other, dim=0)
            spectrograms[..., j] = torch.abs(torch.stft(target_est, self.n_fft, hop_length=self.n_hop, return_complex=True))

        # transposing it as
        # (nb_samples, nb_frames, nb_bins,{1,nb_channels}, nb_sources)

        spectrograms = spectrograms.permute(2, 1, 0, 3)

        # rearranging it into:
        # (nb_samples, nb_frames, nb_bins, nb_channels, 2) to feed
        # into filtering methods
        mix_stft = mix_stft.permute(2, 1, 0, 3)

        nb_frames = spectrograms.shape[0]
        targets_stft = torch.zeros(
            mix_stft.shape + (nb_sources,), dtype=audio.dtype, device=mix_stft.device
        )

        pos = 0
        if self.wiener_win_len:
            wiener_win_len = self.wiener_win_len
        else:
            wiener_win_len = nb_frames
        while pos < nb_frames:
            cur_frame = torch.arange(pos, min(nb_frames, pos + wiener_win_len))
            pos = int(cur_frame[-1]) + 1

            targets_stft[cur_frame] = wiener(
                spectrograms[cur_frame],
                mix_stft[cur_frame],
                self.niter,
                softmask=self.softmask,
                slicq=False, # stft wiener
            )

        # getting to (nb_samples, nb_targets, channel, fft_size, n_frames, 2)
        targets_stft = torch.view_as_complex(targets_stft.permute(4, 2, 1, 0, 3).contiguous())

        # inverse STFT
        estimates = torch.empty(audio.shape + (nb_sources,), dtype=audio.dtype, device=audio.device)

        for j, target_name in enumerate(self.ordered_targets):
            estimates[..., j] = torch.istft(targets_stft[j, ...], self.n_fft, hop_length=self.n_hop, length=audio.shape[-1])

        estimates = torch.unsqueeze(estimates, dim=0).permute(0, 3, 1, 2).contiguous()
        return estimates


    def to_dict(self, estimates: Tensor, aggregate_dict: Optional[dict] = None) -> dict:
        """Convert estimates as stacked tensor to dictionary

        Args:
            estimates (Tensor): separated targets of shape
                (nb_samples, nb_targets, nb_channels, nb_timesteps)
            aggregate_dict (dict or None)

        Returns:
            (dict of str: Tensor):
        """
        estimates_dict = {}
        for k, target in enumerate(["vocals", "drums", "bass", "other"]):
            estimates_dict[target] = estimates[:, k, ...]

        if aggregate_dict is not None:
            new_estimates = {}
            for key in aggregate_dict:
                new_estimates[key] = torch.tensor(0.0)
                for target in aggregate_dict[key]:
                    new_estimates[key] = new_estimates[key] + estimates_dict[target]
            estimates_dict = new_estimates
        return estimates_dict
