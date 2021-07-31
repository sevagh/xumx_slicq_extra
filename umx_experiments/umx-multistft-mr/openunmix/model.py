from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import GRU, BatchNorm2d, Parameter, ReLU, Tanh, Conv2d, ConvTranspose2d, Sigmoid
from .filtering import wiener
from .transforms import make_filterbanks, ComplexNorm


class OpenUnmix(nn.Module):
    """OpenUnmix Core spectrogram based separation module.

    Args:
        nb_bins (int): Number of input time-frequency bins (Default: `2049`).
        nb_channels (int): Number of input audio channels (Default: `2`).
        hidden_size (int): Size for bottleneck layers (Default: `512`).
        nb_layers (int): Number of Bi-RNN layers (Default: `3`).
        unidirectional (bool): Use causal model useful for realtime purpose.
            (Default `False`)
        input_mean (ndarray or None): global data mean of shape `(nb_bins, )`.
            Defaults to zeros(nb_bins)
        input_scale (ndarray or None): global data mean of shape `(nb_bins, )`.
            Defaults to ones(nb_bins)
        max_bin (int or None): Internal frequency bin threshold to
            reduce high frequency content. Defaults to `None` which results
            in `nb_bins`
    """

    def __init__(
        self,
        nb_bins=2049,
        nb_channels=2,
        nb_layers=3,
        unidirectional=False,
        input_mean=None,
        input_scale=None,
        input_mean_big=None,
        input_scale_big=None,
        input_mean_small=None,
        input_scale_small=None,
        max_bin=None,
    ):
        super(OpenUnmix, self).__init__()

        self.nb_output_bins = nb_bins
        if max_bin:
            self.nb_bins = max_bin
        else:
            self.nb_bins = self.nb_output_bins

        self.nb_bins_big = (self.nb_bins-1)*4 + 1
        self.nb_bins_small = (self.nb_bins-1)//4 + 1

        hidden_size = 512
        self.hidden_size = hidden_size

        conv_chans_1 = 12
        conv_chans_2 = 20

        # convolutional layers to combine multiple STFTs
        self.bigwin_layers = torch.nn.ModuleList([
            Conv2d(nb_channels, conv_chans_1, (1, 15), stride=(1, 16)), # downsample frequency to 512, hidden_size
            BatchNorm2d(conv_chans_1),
            Tanh(),
            ConvTranspose2d(conv_chans_1, conv_chans_2, (15, 1), stride=(4, 1)), # upsample time by 4
            BatchNorm2d(conv_chans_2),
            Tanh(),
        ])

        self.midwin_layers = torch.nn.ModuleList([
            Conv2d(nb_channels, conv_chans_2, (1, 5), stride=(1, 4)), # downsample frequency to hidden_size
            BatchNorm2d(conv_chans_2),
            Tanh(),
        ])

        self.smallwin_layers = torch.nn.ModuleList([
            Conv2d(nb_channels, conv_chans_2, (13, 2), stride=(4, 1)), # downsample time by 4
            BatchNorm2d(conv_chans_2),
            Tanh(),
        ])

        # 1x1 convolution layer to reduce the 3x collated spectrograms to 1x
        # this is the "multiresolution" convolution
        self.mr_conv = Conv2d(3*conv_chans_2, 1, 1)
        self.mr_bn = BatchNorm2d(1)
        self.mr_act = ReLU()

        if unidirectional:
            rnn_hidden_size = hidden_size
        else:
            rnn_hidden_size = hidden_size // 2

        self.rnn = GRU(
            input_size=hidden_size,
            hidden_size=rnn_hidden_size,
            num_layers=nb_layers,
            bidirectional=not unidirectional,
            batch_first=False,
            dropout=0.4 if nb_layers > 1 else 0,
        )

        # TODO try bigger channels here
        decoder_chans = 5

        # convolutional decoder on the other side
        self.decoder_layers = torch.nn.ModuleList([
            ConvTranspose2d(1, decoder_chans, (1, 513), stride=1), # upsample frequency to full output dimension in 2 layers
            BatchNorm2d(decoder_chans),
            ReLU(),
            ConvTranspose2d(decoder_chans, nb_channels, (1, 514), stride=1),
            Sigmoid(),
        ])

        if input_mean is not None:
            input_mean = torch.from_numpy(-input_mean[: self.nb_bins]).float()
            input_mean_big = torch.from_numpy(-input_mean_big[: self.nb_bins_big]).float()
            input_mean_small = torch.from_numpy(-input_mean_small[: self.nb_bins_small]).float()
        else:
            input_mean = torch.zeros(self.nb_bins)
            input_mean_big = torch.zeros(self.nb_bins_big)
            input_mean_small = torch.zeros(self.nb_bins_small)

        if input_scale is not None:
            input_scale = torch.from_numpy(1.0 / input_scale[: self.nb_bins]).float()
            input_scale_big = torch.from_numpy(1.0 / input_scale_big[: self.nb_bins_big]).float()
            input_scale_small = torch.from_numpy(1.0 / input_scale_small[: self.nb_bins_small]).float()
        else:
            input_scale = torch.ones(self.nb_bins)
            input_scale_big = torch.ones(self.nb_bins_big)
            input_scale_small = torch.ones(self.nb_bins_small)

        self.input_mean = Parameter(input_mean)
        self.input_scale = Parameter(input_scale)

        self.input_mean_big = Parameter(input_mean_big)
        self.input_scale_big = Parameter(input_scale_big)

        self.input_mean_small = Parameter(input_mean_small)
        self.input_scale_small = Parameter(input_scale_small)

    def freeze(self):
        # set all parameters as not requiring gradient, more RAM-efficient
        # at test time
        for p in self.parameters():
            p.requires_grad = False
        self.eval()

    def forward(self, x: Tensor, x_bigwin: Optional[Tensor] = None, x_smallwin: Optional[Tensor] = None) -> Tensor:
        """
        Args:
            x: input spectrogram of shape
                `(nb_samples, nb_channels, nb_bins, nb_frames)`

        Returns:
            Tensor: filtered spectrogram of shape
                `(nb_samples, nb_channels, nb_bins, nb_frames)`
        """
        #print()

        # permute so that batch is last for rnn
        x = x.permute(3, 0, 1, 2)
        #print(f'x.shape: {x.shape}')

        mix = x.detach().clone()

        # crop
        x = x[..., : self.nb_bins]
        # shift and scale input to mean=0 std=1 (across all bins)
        x = x + self.input_mean
        x = x * self.input_scale

        # get current spectrogram shape
        nb_frames, nb_samples, nb_channels, nb_bins = x.data.shape

        x_bigwin = x_bigwin.permute(3, 0, 1, 2)
        x_smallwin = x_smallwin.permute(3, 0, 1, 2)

        x_bigwin = x_bigwin[..., : self.nb_bins_big]
        x_bigwin = x_bigwin + self.input_mean_big
        x_bigwin = x_bigwin * self.input_scale_big

        x_smallwin = x_smallwin[..., : self.nb_bins_small]
        x_smallwin = x_smallwin + self.input_mean_small
        x_smallwin = x_smallwin * self.input_scale_small

        x = x.permute(1, 2, 0, 3)
        x_bigwin = x_bigwin.permute(1, 2, 0, 3)
        x_smallwin = x_smallwin.permute(1, 2, 0, 3)

        # combine big and small window (double and half) with some convolutions
        # before feeding to the rest of the umx network
        for layer in self.bigwin_layers:
            #print(f'x_bigwin.shape: {x_bigwin.shape}')
            x_bigwin = layer(x_bigwin)
            #print(f'x_bigwin.shape: {x_bigwin.shape}')

        for layer in self.midwin_layers:
            x = layer(x)

        for layer in self.smallwin_layers:
            x_smallwin = layer(x_smallwin)

        #print(f'000 x.shape: {x.shape}')
        #print(f'000 x_bigwin.shape: {x_bigwin.shape}')
        #print(f'000 x_smallwin.shape: {x_smallwin.shape}')

        reduced_time = min(x.shape[-2], min(x_bigwin.shape[-2], x_smallwin.shape[-2]))

        # crop
        x = x[..., : reduced_time, :]
        x_bigwin = x_bigwin[..., : reduced_time, :]
        x_smallwin = x_smallwin[..., : reduced_time, :]

        #print(f'001 x.shape: {x.shape}')
        #print(f'001 x_bigwin.shape: {x_bigwin.shape}')
        #print(f'001 x_smallwin.shape: {x_smallwin.shape}')

        # concatenate the channels (3 stfts * stereo|mono)
        x = torch.cat([x, x_smallwin, x_bigwin], dim=1)

        #x = x.reshape(*x.shape[:2], x.shape[-1]*x.shape[-2])

        #print(f'x.shape: {x.shape}')
        # conv1d to reduce it back down
        x = self.mr_conv(x)
        x = self.mr_act(x)
        x = self.mr_bn(x)

        #print(f'x.shape: {x.shape}')

        x = x.reshape(reduced_time, nb_samples, self.hidden_size)

        # apply 3-layers of stacked LSTM
        rnn_out, _ = self.rnn(x)

        # rnn skip connection
        x = torch.cat([x, rnn_out], -1)
        #print(f'rnn_out.shape: {rnn_out.shape}')

        # convolutional decoder stage
        #print(f'SKIP CON x.shape: {x.shape}')

        x = x.reshape(nb_samples, 1, reduced_time, x.shape[-1])
        #print(f'x.shape: {x.shape}')

        for layer in self.decoder_layers:
            #print(f'0 DECODER: x.shape: {x.shape}')
            x = layer(x)
            #print(f'1 DECODER: x.shape: {x.shape}')

        #print(f'x.shape: {x.shape}')
        x = x.permute(2, 0, 1, 3)
        #print(f'x.shape: {x.shape}')
        #print(f'mix.shape: {mix.shape}')

        mask = torch.ones_like(mix)

        # ignore the bins that won't fucking go away
        mask[ : reduced_time, ...] = x

        # since our output is non-negative, we can apply RELU
        mask = mask * mix

        # permute back to (nb_samples, nb_channels, nb_bins, nb_frames)
        return mask.permute(1, 2, 3, 0)


class Separator(nn.Module):
    """
    Separator class to encapsulate all the stereo filtering
    as a torch Module, to enable end-to-end learning.

    Args:
        targets (dict of str: nn.Module): dictionary of target models
            the spectrogram models to be used by the Separator.
        niter (int): Number of EM steps for refining initial estimates in a
            post-processing stage. Zeroed if only one target is estimated.
            defaults to `1`.
        residual (bool): adds an additional residual target, obtained by
            subtracting the other estimated targets from the mixture,
            before any potential EM post-processing.
            Defaults to `False`.
        wiener_win_len (int or None): The size of the excerpts
            (number of frames) on which to apply filtering
            independently. This means assuming time varying stereo models and
            localization of sources.
            None means not batching but using the whole signal. It comes at the
            price of a much larger memory usage.
        filterbank (str): filterbank implementation method.
            Supported are `['torch', 'asteroid']`. `torch` is about 30% faster
            compared to `asteroid` on large FFT sizes such as 4096. However,
            asteroids stft can be exported to onnx, which makes is practical
            for deployment.
    """

    def __init__(
        self,
        target_models: dict,
        niter: int = 0,
        softmask: bool = False,
        residual: bool = False,
        sample_rate: float = 44100.0,
        n_fft: int = 4096,
        n_hop: int = 1024,
        nb_channels: int = 2,
        wiener_win_len: Optional[int] = 300,
        filterbank: str = "torch",
    ):
        super(Separator, self).__init__()

        # saving parameters
        self.niter = niter
        self.residual = residual
        self.softmask = softmask
        self.wiener_win_len = wiener_win_len

        self.stft, self.istft = make_filterbanks(
            n_fft=n_fft,
            n_hop=n_hop,
            center=True,
            method=filterbank,
            sample_rate=sample_rate,
        )
        self.complexnorm = ComplexNorm(mono=nb_channels == 1)
        self.stft_big, _  = make_filterbanks(
            n_fft=n_fft*4,
            n_hop=n_hop*4,
            center=True,
            method=filterbank,
            sample_rate=sample_rate,
        )
        self.stft_small, _ = make_filterbanks(
            n_fft=n_fft//4,
            n_hop=n_hop//4,
            center=True,
            method=filterbank,
            sample_rate=sample_rate,
        )

        # registering the targets models
        self.target_models = nn.ModuleDict(target_models)
        # adding till https://github.com/pytorch/pytorch/issues/38963
        self.nb_targets = len(self.target_models)
        # get the sample_rate as the sample_rate of the first model
        # (tacitly assume it's the same for all targets)
        self.register_buffer("sample_rate", torch.as_tensor(sample_rate))

    def freeze(self):
        # set all parameters as not requiring gradient, more RAM-efficient
        # at test time
        for p in self.parameters():
            p.requires_grad = False
        self.eval()

    def forward(self, audio: Tensor) -> Tensor:
        """Performing the separation on audio input

        Args:
            audio (Tensor): [shape=(nb_samples, nb_channels, nb_timesteps)]
                mixture audio waveform

        Returns:
            Tensor: stacked tensor of separated waveforms
                shape `(nb_samples, nb_targets, nb_channels, nb_timesteps)`
        """

        nb_sources = self.nb_targets
        nb_samples = audio.shape[0]

        # getting the STFT of mix:
        # (nb_samples, nb_channels, nb_bins, nb_frames, 2)
        mix_stft = self.stft(audio)
        X_big = self.complexnorm(self.stft_big(audio))
        X_small = self.complexnorm(self.stft_small(audio))
        X = self.complexnorm(mix_stft)

        # initializing spectrograms variable
        spectrograms = torch.zeros(X.shape + (nb_sources,), dtype=audio.dtype, device=X.device)

        for j, (target_name, target_module) in enumerate(self.target_models.items()):
            # apply current model to get the source spectrogram
            target_spectrogram = target_module(X.detach().clone(), X_big.detach().clone(), X_small.detach().clone())
            spectrograms[..., j] = target_spectrogram

        # transposing it as
        # (nb_samples, nb_frames, nb_bins,{1,nb_channels}, nb_sources)
        spectrograms = spectrograms.permute(0, 3, 2, 1, 4)

        # rearranging it into:
        # (nb_samples, nb_frames, nb_bins, nb_channels, 2) to feed
        # into filtering methods
        mix_stft = mix_stft.permute(0, 3, 2, 1, 4)

        # create an additional target if we need to build a residual
        if self.residual:
            # we add an additional target
            nb_sources += 1

        if nb_sources == 1 and self.niter > 0:
            raise Exception(
                "Cannot use EM if only one target is estimated."
                "Provide two targets or create an additional "
                "one with `--residual`"
            )

        nb_frames = spectrograms.shape[1]
        targets_stft = torch.zeros(
            mix_stft.shape + (nb_sources,), dtype=audio.dtype, device=mix_stft.device
        )
        for sample in range(nb_samples):
            pos = 0
            if self.wiener_win_len:
                wiener_win_len = self.wiener_win_len
            else:
                wiener_win_len = nb_frames
            while pos < nb_frames:
                cur_frame = torch.arange(pos, min(nb_frames, pos + wiener_win_len))
                pos = int(cur_frame[-1]) + 1

                targets_stft[sample, cur_frame] = wiener(
                    spectrograms[sample, cur_frame],
                    mix_stft[sample, cur_frame],
                    self.niter,
                    softmask=self.softmask,
                    residual=self.residual,
                )

        # getting to (nb_samples, nb_targets, channel, fft_size, n_frames, 2)
        targets_stft = targets_stft.permute(0, 5, 3, 2, 1, 4).contiguous()

        # inverse STFT
        estimates = self.istft(targets_stft, length=audio.shape[2])

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
        for k, target in enumerate(self.target_models):
            estimates_dict[target] = estimates[:, k, ...]

        # in the case of residual, we added another source
        if self.residual:
            estimates_dict["residual"] = estimates[:, -1, ...]

        if aggregate_dict is not None:
            new_estimates = {}
            for key in aggregate_dict:
                new_estimates[key] = torch.tensor(0.0)
                for target in aggregate_dict[key]:
                    new_estimates[key] = new_estimates[key] + estimates_dict[target]
            estimates_dict = new_estimates
        return estimates_dict
