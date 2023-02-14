from typing import Optional, Union
import sys
from tqdm import trange
from pathlib import Path
import torch
import json
from torch import Tensor
import torch.nn as nn
from .models import Unmix, RefinedUnmix
import norbert
from .transforms import (
    TorchSTFT,
    TorchISTFT,
    ComplexNorm,
    NSGTBase,
    make_filterbanks,
)


"""
Config 0: no stft-wiener-em
Config 1: stft-wiener-em
"""
class Separator(nn.Module):
    @classmethod
    def load(
        cls,
        xumx_config: int,
        device: Union[str, torch.device] = "cpu",
    ):
        model_path = ""
        do_stft_wiener_em = True

        if xumx_config == 0:
            model_path = "/xumx-sliCQ-V2/pretrained_model"
            do_stft_wiener_em = False
        elif xumx_config == 1:
            model_path = "/xumx-sliCQ-V2/pretrained_model"

        model_path = Path(model_path)

        # when path exists, we assume its a custom model saved locally
        assert model_path.exists()

        with open(Path(model_path, "separator.json"), "r") as stream:
            enc_conf = json.load(stream)

        xumx_model = _load_target_models(
            model_path,
            sample_rate=enc_conf["sample_rate"],
            device=device,
        )

        separator = Separator(
            xumx_model=xumx_model,
            do_stft_wiener_em=do_stft_wiener_em,
            sample_rate=enc_conf["sample_rate"],
        ).to(device)

        separator.freeze()
        return separator

    def __init__(
        self,
        xumx_model: Union[Unmix, RefinedUnmix] = None,
        do_stft_wiener_em: bool = True,
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
        self.register_buffer("sample_rate", torch.as_tensor(sample_rate))
        self.chunk_size = chunk_size if chunk_size is not None else sys.maxsize

        self.xumx_model = xumx_model
        self.do_stft_wiener_em = do_stft_wiener_em

        # can skip wiener to evaluate raw xumx-sliCQ-V2 output
        if self.do_stft_wiener_em:
            # Norbert MWF + iSTFT/STFT will be done here
            # for the raw xumx_slicq_v2 model
            self.n_fft = n_fft
            self.n_hop = n_hop
            self.wiener_win_len = wiener_win_len
            self.niter = niter
            self.softmask = softmask

            self.stft = TorchSTFT(self.n_fft, self.n_hop, center=True)
            self.istft = TorchISTFT(self.n_fft, self.n_hop, center=True)
            self.cnorm = ComplexNorm()

    def freeze(self):
        # set all parameters as not requiring gradient, more RAM-efficient
        # at test time
        for p in self.parameters():
            # p.requires_grad = False
            p.grad = None
        self.xumx_model.freeze()
        self.eval()

    @torch.no_grad()
    def post_wiener_stft(self, x, y_all):
        n_samples = x.shape[-1]
        mix_stft = self.stft(x)

        # initializing spectrograms variable
        spectrograms = torch.zeros(
            (4,) + mix_stft.shape[:-1], dtype=mix_stft.dtype, device=mix_stft.device
        )

        spectrograms = self.cnorm(self.stft(y_all))

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

            if self.do_stft_wiener_em:
                # wiener MWF
                print("performing post-Wiener EM with STFTs")
                estimates = self.post_wiener_stft(audio, estimates)
            else:
                print("user requested to skip STFT Wiener-EM...")

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


def _load_target_models(
        model_path: str, device="cpu", sample_rate=44100,
):
    model_path = Path(model_path).expanduser()

    # load model from disk
    with open(Path(model_path, "xumx_slicq_v2.json"), "r") as stream:
        results = json.load(stream)

    # need to configure an NSGT object to peek at its params to set up the neural network
    # e.g. M depends on the sllen which depends on fscale+fmin+fmax
    nsgt_base = NSGTBase(
        results["args"]["fscale"],
        results["args"]["fbins"],
        results["args"]["fmin"],
        fs=sample_rate,
        device=device,
    )

    nb_channels = 2

    seq_dur = results["args"]["seq_dur"]

    target_model_path = Path(model_path, "xumx_slicq_v2.pth")
    state = torch.load(target_model_path, map_location=device)

    jagged_slicq, _ = nsgt_base.predict_input_size(1, nb_channels, seq_dur)
    cnorm = ComplexNorm().to(device)

    nsgt, insgt = make_filterbanks(
        nsgt_base, sample_rate
    )
    nsgt = nsgt.to(device)
    insgt = insgt.to(device)

    jagged_slicq_cnorm = cnorm(jagged_slicq)

    xumx_model = Unmix(
        jagged_slicq_cnorm,
        (nsgt, insgt, cnorm),
        max_bin=nsgt_base.max_bins(results["args"]["bandwidth"]),
    )

    xumx_model.load_state_dict(state, strict=False)
    xumx_model.freeze()
    xumx_model.to(device)

    return xumx_model
