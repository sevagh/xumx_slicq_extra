from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter, ReLU, Tanh, BatchNorm2d, ConvTranspose2d, Conv2d, Sequential, Sigmoid, ModuleList, Linear
from .filtering import atan2, wiener
from .transforms import make_filterbanks, ComplexNorm, phasemix_sep, NSGTBase, overlap_add_slicq, repeated_interpolation, repeated_deinterpolation
from collections import defaultdict
import numpy as np
import copy

eps = 1.e-10


class OpenUnmixCDAE_B(nn.Module):
    def __init__(
        self,
        slicq_sample_input,
        ragged_shapes,
        max_bin=None,
        input_mean=None,
        input_scale=None,
    ):
        super(OpenUnmixCDAE_B, self).__init__()

        nb_samples, nb_channels, nb_f_bins, nb_t_bins = slicq_sample_input.shape
        self.max_bin = max_bin

        channels = [nb_channels, 12, 24, 48]
        layers = len(channels)-1

        filters = [(3, 5)]*layers

        encoder = []
        decoder = []

        layers = len(filters)

        for i in range(layers):
            encoder.append(
                Sequential(
                    Conv2d(channels[i], channels[i+1], filters[i], bias=False),
                    BatchNorm2d(channels[i+1]),
                    ReLU(),
                )
            )

        for i in range(layers,0,-1):
            if i == 1:
                decoder.append(
                    Sequential(
                        ConvTranspose2d(channels[i], channels[i-1], filters[i-1], bias=True),
                        Sigmoid(),
                    )
                )
            else:
                decoder.append(
                    Sequential(
                        ConvTranspose2d(channels[i], channels[i-1], filters[i-1], bias=False),
                        BatchNorm2d(channels[i-1]),
                        ReLU()
                    )
                )

        self.cdae = Sequential(*encoder, *decoder)

        # deoverlap
        deoverlap_layers = [None]*len(ragged_shapes)

        for i, ragged_shape in enumerate(ragged_shapes):
            nwin = ragged_shape[-1]
            deoverlap_layers[i] = Linear(in_features=nwin, out_features=nwin, bias=True)

        self.deoverlap_layers = ModuleList(deoverlap_layers)

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
            #p.requires_grad = False
            p.grad = None
        self.eval()

    def forward(self, x: Tensor) -> Tensor:
        ragged_shapes = [x_.shape for x_ in x]
        ola = overlap_add_slicq(x)
        ragged_shapes_ola = [ola_.shape for ola_ in ola]
        x = repeated_interpolation(ola)

        mix = x.detach().clone()

        # shift and scale input to mean=0 std=1 (across all bins)
        x = x.permute(0, 1, 3, 2)
        x += self.input_mean
        x *= self.input_scale
        x = x.permute(0, 1, 3, 2)

        x_shape = x.shape
        nb_samples, nb_channels, nb_f_bins, nb_t_bins = x_shape

        # only pass up to max_bins through the CDAE
        tmp = x[..., : self.max_bin, :].detach().clone()
        for i, layer in enumerate(self.cdae):
            tmp = layer(tmp)

        x[..., : self.max_bin, :] = tmp

        # multiplicative skip connection
        x = x * mix

        x = repeated_deinterpolation(x, ragged_shapes_ola)
        x = self.deoverlap(x, ragged_shapes)

        return x

    def deoverlap(self, x, ragged_shapes) -> Tensor:
        ret = [None]*len(x)
        for i, ragged_shape in enumerate(ragged_shapes):
            nwin = ragged_shape[-1]
            nb_slices = ragged_shape[-2]

            hop = nwin//2
            nb_m_bins = nwin

            nb_samples, nb_channels, nb_f_bins, ncoefs = x[i].shape

            out = torch.zeros((nb_samples, nb_channels, nb_f_bins, nb_slices, nwin), dtype=x[i].dtype, device=x[i].device)

            # each slice considers nwin coefficients
            ptr = 0
            for j in range(nb_slices):
                # inverse of overlap-add
                out[:, :, :, j, :] = self.deoverlap_layers[i](x[i][:, :, :, ptr:ptr+nwin])
                ptr += hop

            ret[i] = out

        return ret



class OpenUnmixCDAE(nn.Module):
    def __init__(
        self,
        slicq_sample_input,
        ragged_shapes,
        max_bin=None,
        input_mean=None,
        input_scale=None,
    ):
        super(OpenUnmixCDAE, self).__init__()

        nb_samples, nb_channels, nb_f_bins, nb_t_bins = slicq_sample_input.shape
        self.max_bin = max_bin

        channels = [nb_channels, 25, 55]
        layers = len(channels)-1

        freq_filter = 5
        time_filter = 13

        filters = [(freq_filter, time_filter)]*layers

        encoder = []
        decoder = []

        layers = len(filters)

        for i in range(layers):
            encoder.append(
                Sequential(
                    Conv2d(channels[i], channels[i+1], filters[i], dilation=(1,2), bias=False),
                    BatchNorm2d(channels[i+1]),
                    ReLU(),
                )
            )

        for i in range(layers,0,-1):
            if i == 1:
                decoder.append(
                    Sequential(
                        ConvTranspose2d(channels[i], channels[i-1], filters[i-1], dilation=(1,2), bias=True),
                        Sigmoid(),
                    )
                )
            else:
                decoder.append(
                    Sequential(
                        ConvTranspose2d(channels[i], channels[i-1], filters[i-1], dilation=(1,2), bias=False),
                        BatchNorm2d(channels[i-1]),
                        ReLU()
                    )
                )

        self.cdae = Sequential(*encoder, *decoder)

        # deoverlap
        deoverlap_layers = [None]*len(ragged_shapes)

        for i, ragged_shape in enumerate(ragged_shapes):
            nwin = ragged_shape[-1]
            deoverlap_layers[i] = Linear(in_features=nwin, out_features=nwin, bias=True)

        self.deoverlap_layers = ModuleList(deoverlap_layers)

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
            #p.requires_grad = False
            p.grad = None
        self.eval()

    def forward(self, x: Tensor) -> Tensor:
        ragged_shapes = [x_.shape for x_ in x]
        ola = overlap_add_slicq(x)
        ragged_shapes_ola = [ola_.shape for ola_ in ola]
        x = repeated_interpolation(ola)

        mix = x.detach().clone()

        # shift and scale input to mean=0 std=1 (across all bins)
        x = x.permute(0, 1, 3, 2)
        x += self.input_mean
        x *= self.input_scale
        x = x.permute(0, 1, 3, 2)

        x_shape = x.shape
        nb_samples, nb_channels, nb_f_bins, nb_t_bins = x_shape

        # only pass up to max_bins through the CDAE
        tmp = x[..., : self.max_bin, :].detach().clone()
        for i, layer in enumerate(self.cdae):
            tmp = layer(tmp)

        x[..., : self.max_bin, :] = tmp

        # multiplicative skip connection
        x = x * mix

        x = repeated_deinterpolation(x, ragged_shapes_ola)
        x = self.deoverlap(x, ragged_shapes)

        return x

    def deoverlap(self, x, ragged_shapes) -> Tensor:
        ret = [None]*len(x)
        for i, ragged_shape in enumerate(ragged_shapes):
            nwin = ragged_shape[-1]
            nb_slices = ragged_shape[-2]

            hop = nwin//2
            nb_m_bins = nwin

            nb_samples, nb_channels, nb_f_bins, ncoefs = x[i].shape

            out = torch.zeros((nb_samples, nb_channels, nb_f_bins, nb_slices, nwin), dtype=x[i].dtype, device=x[i].device)

            # each slice considers nwin coefficients
            ptr = 0
            for j in range(nb_slices):
                # inverse of overlap-add
                out[:, :, :, j, :] = self.deoverlap_layers[i](x[i][:, :, :, ptr:ptr+nwin])
                ptr += hop

            ret[i] = out

        return ret


class OpenUnmix(nn.Module):
    """OpenUnmix Core spectrogram based separation module.

    Args:
        nb_bins, M (int): Number of sliCQ-NSGT tf bins
        nb_channels (int): Number of input audio channels (Default: `2`).
    """
    def __init__(
        self,
        jagged_slicq_sample_input,
        model_b=False,
        max_bin=None,
        input_mean=None,
        input_scale=None,
    ):
        super(OpenUnmix, self).__init__()

        ragged_shapes = [jagged_slicq_sample_input_.shape for jagged_slicq_sample_input_ in jagged_slicq_sample_input]

        jagged_slicq_interp = repeated_interpolation(overlap_add_slicq(jagged_slicq_sample_input))

        if model_b:
            self.unmix_vocals = OpenUnmixCDAE_B(
                jagged_slicq_interp,
                ragged_shapes,
                max_bin=max_bin,
                input_mean=input_mean,
                input_scale=input_scale,
            )
        else:
            self.unmix_vocals = OpenUnmixCDAE(
                jagged_slicq_interp,
                ragged_shapes,
                max_bin=max_bin,
                input_mean=input_mean,
                input_scale=input_scale,
            )

        self.unmix_bass = copy.deepcopy(self.unmix_vocals)
        self.unmix_drums = copy.deepcopy(self.unmix_vocals)
        self.unmix_other = copy.deepcopy(self.unmix_vocals)

    def freeze(self):
        # set all parameters as not requiring gradient, more RAM-efficient
        # at test time
        for p in self.parameters():
            #p.requires_grad = False
            p.grad = None
        self.eval()

    def forward(self, x, vocals=True, bass=True, drums=True, other=True) -> Tensor:
        y_vocals = self.unmix_vocals(x) if vocals else None
        y_bass = self.unmix_bass(x) if bass else None
        y_drums = self.unmix_drums(x) if drums else None
        y_other = self.unmix_other(x) if other else None

        return y_bass, y_vocals, y_other, y_drums


class Separator(nn.Module):
    def __init__(
        self,
        xumx_model,
        xumx_nsgt,
        jagged_slicq_sample_input,
        sample_rate: float = 44100.0,
        nb_channels: int = 2,
        device: str = "cpu",
        niter: int = 1,
        stft_wiener: bool = True,
        softmask: bool = False,
        wiener_win_len: Optional[int] = 300,
        n_fft: Optional[int] = 4096,
        n_hop: Optional[int] = 1024,
    ):
        super(Separator, self).__init__()
        self.stft_wiener = stft_wiener

        # saving parameters
        self.niter = niter
        self.softmask = softmask

        self.device = device

        self.nsgt, self.insgt = make_filterbanks(
            xumx_nsgt, sample_rate=sample_rate
        )

        self.complexnorm = ComplexNorm(mono=nb_channels == 1)
        self.nb_channels = nb_channels

        self.xumx_model = xumx_model
        self.register_buffer("sample_rate", torch.as_tensor(sample_rate))

        self.n_fft = n_fft
        self.n_hop = n_hop
        self.wiener_win_len = wiener_win_len

        if not self.stft_wiener:
            # first, get frequency and time limits to build the large zero-padded matrix
            total_f_bins = 0
            max_t_bins = 0
            for i, block in enumerate(jagged_slicq_sample_input):
                nb_samples, nb_channels, nb_f_bins, nb_slices, nb_t_bins = block.shape
                total_f_bins += nb_f_bins
                max_t_bins = max(max_t_bins, nb_t_bins)

            self.total_f_bins = total_f_bins
            self.max_t_bins = max_t_bins

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

        X = self.nsgt(audio)
        Xmag = self.complexnorm(X)

        # xumx inference - magnitude slicq estimate
        Ymag_bass, Ymag_vocals, Ymag_other, Ymag_drums = self.xumx_model(Xmag)

        if self.stft_wiener:
            print('STFT WIENER')

            # initial mix phase + magnitude estimate
            Ycomplex_bass = phasemix_sep(X, Ymag_bass)
            Ycomplex_vocals = phasemix_sep(X, Ymag_vocals)
            Ycomplex_drums = phasemix_sep(X, Ymag_drums)
            Ycomplex_other = phasemix_sep(X, Ymag_other)

            y_bass = self.insgt(Ycomplex_bass, audio.shape[-1])
            y_drums = self.insgt(Ycomplex_drums, audio.shape[-1])
            y_other = self.insgt(Ycomplex_other, audio.shape[-1])
            y_vocals = self.insgt(Ycomplex_vocals, audio.shape[-1])

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
        else:
            print('sliCQT WIENER')

            # block-wise wiener
            # assemble it all into a zero-padded matrix

            nb_slices = X[0].shape[3]
            last_dim = 2

            X_matrix = torch.zeros((nb_samples, self.nb_channels, self.total_f_bins, nb_slices, self.max_t_bins, last_dim), dtype=X[0].dtype, device=X[0].device)
            spectrograms = torch.zeros(X_matrix.shape[:-1] + (nb_sources,), dtype=audio.dtype, device=X_matrix.device)

            freq_start = 0
            for i, X_block in enumerate(X):
                nb_samples, self.nb_channels, nb_f_bins, nb_slices, nb_t_bins, last_dim = X_block.shape

                # assign up to the defined time bins - to the right will be zeros
                X_matrix[:, :, freq_start:freq_start+nb_f_bins, :, : nb_t_bins, :] = X_block

                spectrograms[:, :, freq_start:freq_start+nb_f_bins, :, : nb_t_bins, 0] = Ymag_vocals[i]
                spectrograms[:, :, freq_start:freq_start+nb_f_bins, :, : nb_t_bins, 1] = Ymag_drums[i]
                spectrograms[:, :, freq_start:freq_start+nb_f_bins, :, : nb_t_bins, 2] = Ymag_bass[i]
                spectrograms[:, :, freq_start:freq_start+nb_f_bins, :, : nb_t_bins, 3] = Ymag_other[i]

                freq_start += nb_f_bins

            spectrograms = wiener(
                torch.squeeze(spectrograms, dim=0),
                torch.squeeze(X_matrix, dim=0),
                self.niter,
                softmask=self.softmask,
                slicq=True,
            )

            # reverse the wiener/EM permutes etc.
            spectrograms = torch.unsqueeze(spectrograms.permute(2, 1, 0, 3, 4), dim=0)
            spectrograms = spectrograms.reshape(nb_samples, self.nb_channels, self.total_f_bins, nb_slices, self.max_t_bins, *spectrograms.shape[-2:])

            slicq_vocals = [None]*len(X)
            slicq_bass = [None]*len(X)
            slicq_drums = [None]*len(X)
            slicq_other = [None]*len(X)

            estimates = torch.empty(audio.shape + (nb_sources,), dtype=audio.dtype, device=audio.device)

            nb_samples, self.nb_channels, nb_f_bins, nb_slices, nb_t_bins = X_matrix.shape[:-1]

            # matrix back to list form for insgt
            freq_start = 0
            for i, X_block in enumerate(X):
                nb_samples, self.nb_channels, nb_f_bins, nb_slices, nb_t_bins, _ = X_block.shape

                slicq_vocals[i] = spectrograms[:, :, freq_start:freq_start+nb_f_bins, :, : nb_t_bins, :, 0].contiguous()
                slicq_drums[i] = spectrograms[:, :, freq_start:freq_start+nb_f_bins, :, : nb_t_bins, :, 1].contiguous()
                slicq_bass[i] = spectrograms[:, :, freq_start:freq_start+nb_f_bins, :, : nb_t_bins, :, 2].contiguous()
                slicq_other[i] = spectrograms[:, :, freq_start:freq_start+nb_f_bins, :, : nb_t_bins, :, 3].contiguous()

                freq_start += nb_f_bins

            estimates[..., 0] = self.insgt(slicq_vocals, audio.shape[-1])
            estimates[..., 1] = self.insgt(slicq_drums, audio.shape[-1])
            estimates[..., 2] = self.insgt(slicq_bass, audio.shape[-1])
            estimates[..., 3] = self.insgt(slicq_other, audio.shape[-1])

            estimates = estimates.permute(0, 3, 1, 2).contiguous()

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
