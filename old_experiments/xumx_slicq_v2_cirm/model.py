from typing import Optional
from tqdm import trange
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import norbert
from .transforms import (
    make_filterbanks,
    NSGTBase,
    TorchSTFT,
    TorchISTFT,
)
from .target_model import UnmixAllTargets
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

        self.umx = UnmixAllTargets(jagged_slicq_sample_input, max_bin, input_means, input_scales)

        self.nsgt, self.insgt = encoder

    def freeze(self):
        # set all parameters as not requiring gradient, more RAM-efficient
        # at test time
        for p in self.parameters():
            # p.requires_grad = False
            p.grad = None
        self.eval()

    def forward(self, x, return_nsgts: bool =False) -> Tensor:
        n_samples = x.shape[-1]

        X = self.nsgt(x)
        Ycomplex_all = self.umx(X)

        y_all = self.insgt(Ycomplex_all, n_samples)
        #print(f"y_all: {y_all.shape}")
        if return_nsgts:
            return y_all, Ycomplex_all
        return y_all


class Separator(nn.Module):
    def __init__(
        self,
        xumx_model,
        sample_rate: float = 44100.0,
        device: str = "cpu",
        chunk_size: Optional[int] = 2621440,
        wiener_win_len: Optional[int] = 300,
        niter: int = 1,
        softmask: bool = False,
        n_fft: Optional[int] = 4096,
        n_hop: Optional[int] = 1024,
    ):
        super(Separator, self).__init__()
        # saving parameters

        self.device = device
        self.nb_channels = 2
        self.xumx_model = xumx_model
        self.register_buffer("sample_rate", torch.as_tensor(sample_rate))
        self.chunk_size = chunk_size if chunk_size is not None else sys.maxsize

        # Norbert MWF + iSTFT/STFT will be done here
        self.n_fft = n_fft
        self.n_hop = n_hop
        self.wiener_win_len = wiener_win_len
        self.niter = niter
        self.softmask = softmask

        self.stft = TorchSTFT(self.n_fft, self.n_hop, center=True)
        self.istft = TorchISTFT(self.n_fft, self.n_hop, center=True)

    def freeze(self):
        # set all parameters as not requiring gradient, more RAM-efficient
        # at test time
        for p in self.parameters():
            # p.requires_grad = False
            p.grad = None
        self.xumx_model.freeze()
        self.eval()

    @torch.no_grad()
    def post_wiener(self, x, y_all):
        n_samples = x.shape[-1]
        mix_stft = self.stft(x)

        # initializing spectrograms variable
        spectrograms = torch.zeros(
            (4,) + mix_stft.shape[:-1], dtype=mix_stft.dtype, device=mix_stft.device
        )

        spectrograms = torch.abs(torch.view_as_complex(self.stft(y_all)))

        # transposing it as
        # (nb_samples, nb_frames, nb_bins,{1,nb_channels}, nb_sources)
        spectrograms = spectrograms.permute(1, 4, 3, 2, 0)

        # rearranging it into:
        # (nb_samples, nb_frames, nb_bins, nb_channels, 2) to feed
        # into filtering methods
        mix_stft = mix_stft.permute(0, 3, 2, 1, 4)

        nb_frames = spectrograms.shape[1]
        targets_stft = torch.zeros(
            *mix_stft.shape[:-1] + (4,2,),
            dtype=mix_stft.dtype,
            device=mix_stft.device,
        )

        pos = 0
        if self.wiener_win_len:
            wiener_win_len = self.wiener_win_len
        else:
            wiener_win_len = nb_frames
        while pos < nb_frames:
            cur_frame = torch.arange(pos, min(nb_frames, pos + wiener_win_len))
            pos = int(cur_frame[-1]) + 1

            targets_stft[:, cur_frame, ...] = torch.view_as_real(norbert.wiener(
                spectrograms[:, cur_frame, ...],
                torch.view_as_complex(mix_stft[:, cur_frame, ...]),
                self.niter,
                use_softmask=self.softmask,
            ))

        # getting to (nb_samples, nb_targets, channel, fft_size, n_frames)
        targets_stft = targets_stft.permute(4, 0, 3, 2, 1, 5).contiguous()

        # inverse STFT
        estimates = torch.empty(
            (4,) + x.shape, dtype=x.dtype, device=x.device
        )

        estimates = self.istft(targets_stft, length=n_samples)
        #print(f"ests: {estimates.shape}")

        return estimates

    @torch.no_grad()
    def forward(self, audio_big: Tensor) -> Tensor:
        """Performing the separation on audio input

        Args:
            audio (Tensor): [shape=(nb_samples, nb_channels, nb_timesteps)]
                mixture audio waveform

        Returns:
            Tensor: stacked tensor of separated waveforms
                shape `(nb_samples, nb_targets, nb_channels, nb_timesteps)`
        """
        nb_samples = audio_big.shape[0]
        N = audio_big.shape[-1]

        nchunks = N // self.chunk_size
        if (N % self.chunk_size) != 0:
            nchunks += 1

        print(f"n chunks: {nchunks}")

        final_estimates = []

        for chunk_idx in trange(nchunks):
            audio = audio_big[
                ...,
                chunk_idx * self.chunk_size : min((chunk_idx + 1) * self.chunk_size, N),
            ]
            print(f"audio.shape: {audio.shape}")

            # xumx inference
            estimates = self.xumx_model(audio)

            # wiener MWF
            estimates = self.post_wiener(audio, estimates)

            final_estimates.append(estimates)

        ests_concat = torch.cat(final_estimates, axis=-1)
        print(f"ests concat: {ests_concat.shape}")
        return ests_concat

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

        # follow the ordering in data.py
        for k, target in enumerate(["bass", "vocals", "other", "drums"]):
            estimates_dict[target] = estimates[k]

        if aggregate_dict is not None:
            new_estimates = {}
            for key in aggregate_dict:
                new_estimates[key] = torch.tensor(0.0)
                for target in aggregate_dict[key]:
                    new_estimates[key] = new_estimates[key] + estimates_dict[target]
            estimates_dict = new_estimates
        return estimates_dict
