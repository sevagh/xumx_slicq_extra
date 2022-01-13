from typing import Optional
import numpy as np
import torch
import torchaudio
from torch import Tensor
import torch.nn as nn
from .filtering import atan2
import warnings
from typing import List, Tuple
from .nsgt import NSGT_sliced, BarkScale, MelScale, LogScale, VQLogScale, OctScale
from torch.nn.functional import interpolate


def phasemix_sep(X, Ymag):
    Ycomplex = [None]*len(X)

    for i, X_block in enumerate(X):
        Xphase_block = atan2(X_block[..., 1], X_block[..., 0])
        Ycomplex_block = torch.empty_like(X_block)

        Ycomplex_block[..., 0] = Ymag[i] * torch.cos(Xphase_block)
        Ycomplex_block[..., 1] = Ymag[i] * torch.sin(Xphase_block)

        Ycomplex[i] = Ycomplex_block

    return Ycomplex


def make_filterbanks_slicqt(nsgt_base, sample_rate=44100.0):
    if sample_rate != 44100.0:
        raise ValueError('i was lazy and harcoded a lot of 44100.0, forgive me')

    encoder = NSGT_SL(nsgt_base)
    decoder = INSGT_SL(nsgt_base)

    return encoder, decoder


class NSGTBase(nn.Module):
    def __init__(self, *,
        scale=None,
        fbins=None,
        fmin=None,
        fs=44100,
        device="cuda",
        gamma=25.,
    ):
        super(NSGTBase, self).__init__()

        if scale is None or fbins is None or fmin is None:
            raise ValueError('please instantiate NSGTBase with scale=, fbins=, fmin= explicitly')

        self.fbins = fbins
        self.fmin = fmin
        self.gamma = gamma
        self.fmax = fs/2

        self.scl = None
        if scale == 'bark':
            self.scl = BarkScale(self.fmin, self.fmax, self.fbins)
        elif scale == 'mel':
            self.scl = MelScale(self.fmin, self.fmax, self.fbins)
        elif scale == 'cqlog':
            self.scl = LogScale(self.fmin, self.fmax, self.fbins)
        elif scale == 'vqlog':
            self.scl = VQLogScale(self.fmin, self.fmax, self.fbins, self.gamma)
        elif scale == 'oct':
            self.scl = OctScale(self.fmin, self.fmax, self.fbins)
        else:
            raise ValueError(f'unsupported frequency scale {scale}')

        self.sllen, self.trlen = self.scl.suggested_sllen_trlen(fs)

        self.freqs, self.q_factors = self.scl()

        self.nsgt = NSGT_sliced(self.scl, self.sllen, self.trlen, fs, real=True, multichannel=True, device=device)
        self.M = self.nsgt.ncoefs
        self.fs = fs
        self.fbins_actual = self.nsgt.fbins_actual

    def predict_input_size(self, batch_size, nb_channels, seq_dur_s):
        fwd = NSGT_SL(self)

        x = torch.rand((batch_size, nb_channels, int(seq_dur_s*self.fs)), dtype=torch.float32)
        shape = x.size()
        nb_samples, nb_channels, nb_timesteps = shape

        nsgt_f = fwd(x)
        return nsgt_f

    def _apply(self, fn):
        self.nsgt._apply(fn)
        return self


class NSGT_SL(nn.Module):
    def __init__(self, nsgt):
        super(NSGT_SL, self).__init__()
        self.nsgt = nsgt

    def _apply(self, fn):
        self.nsgt._apply(fn)
        return self

    def forward(self, x: Tensor) -> Tensor:
        """NSGT forward path
        Args:
            x (Tensor): audio waveform of
                shape (nb_samples, nb_channels, nb_timesteps)
        Returns:
            NSGT (Tensor): complex nsgt of
                shape (nb_samples, nb_channels, nb_bins_1, nb_bins_2, nb_frames, complex=2)
                last axis is stacked real and imaginary
        """
        shape = x.size()
        nb_samples, nb_channels, nb_timesteps = shape

        # pack batch
        x = x.view(-1, shape[-1])

        C = self.nsgt.nsgt.forward((x,))

        for i, nsgt_f in enumerate(C):
            nsgt_f = torch.moveaxis(nsgt_f, 0, -2)
            nsgt_f = torch.view_as_real(nsgt_f)
            # unpack batch
            nsgt_f = nsgt_f.view(shape[:-1] + nsgt_f.shape[-4:])
            C[i] = nsgt_f

        return C

    def interpolate(self, x: Tensor) -> Tensor:
        max_time = self.nsgt.M
        nfbins = self.nsgt.fbins_actual

        interpolated = torch.zeros((*x[0].shape[:-3], nfbins, x[0].shape[-2], max_time), dtype=x[0].dtype, device=x[0].device)

        fbin_ptr = 0
        for i, x_bucket in enumerate(x):
            freqs = x_bucket.shape[-3]
            slices = x_bucket.shape[-2]

            if x_bucket.shape[-1] == max_time:
                # same time width, no interpolation
                interpolated[:, :, fbin_ptr:fbin_ptr+freqs, :] = x_bucket
            else:
                interpolated[:, :, fbin_ptr:fbin_ptr+freqs, :] = interpolate(x_bucket, size=(freqs, slices, max_time), mode='trilinear', align_corners=False)

            fbin_ptr += freqs

        return interpolated

    def deinterpolate(self, interpolated, ragged_shapes):
        max_time = interpolated.shape[-1]
        full_slicq = []
        fbin_ptr = 0
        for i, bucket_shape in enumerate(ragged_shapes):
            curr_slicq = torch.zeros(bucket_shape, dtype=interpolated.dtype, device=interpolated.device)

            small_time = bucket_shape[-1]
            slices = bucket_shape[-2]
            freqs = bucket_shape[-3]

            if bucket_shape[-1] == interpolated.shape[-1]:
                # same time width, no interpolation
                curr_slicq = interpolated[:, :, fbin_ptr:fbin_ptr+freqs, :]
            else:
                curr_slicq = interpolate(interpolated[:, :, fbin_ptr:fbin_ptr+freqs, :], size=(freqs, slices, small_time), mode='trilinear', align_corners=False)

            full_slicq.append(curr_slicq)

            fbin_ptr += freqs
        return full_slicq

    # forward only; learned inverse operation is in deoverlapnet
    def overlap_add(self, slicq):
        nb_samples, nb_channels, nb_f_bins, nb_slices, nb_m_bins = slicq.shape

        nwin = nb_m_bins

        ncoefs = ((1+nb_slices)*nb_m_bins)//2

        hop = nwin//2 # 50% overlap window

        out = torch.zeros((nb_samples, nb_channels, nb_f_bins, ncoefs), dtype=slicq.dtype, device=slicq.device)

        ptr = 0

        for i in range(nb_slices):
            # weighted overlap-add with last `hop` samples
            # rectangular window
            out[:, :, :, ptr:ptr+nwin] += slicq[:, :, :, i, :]
            ptr += hop

        return out

    def plot_spectrogram(self, mls, ax):
        assert mls.shape[0] == 1
        # remove batch
        mls = torch.squeeze(mls, dim=0)
        # mix down multichannel
        mls = torch.mean(mls, dim=-1)
        fs_coef = self.nsgt.fs*self.nsgt.nsgt.coef_factor
        mls_dur = len(mls)/fs_coef # final duration of MLS
        mls_max = torch.quantile(mls, 0.9)

        mls = mls.detach().cpu().numpy()
        ax.imshow(mls.T, aspect=mls_dur/mls.shape[1]*0.2, interpolation='nearest', origin='lower', vmin=mls_max-60., vmax=mls_max, extent=(0,mls_dur,0,mls.shape[1]))


class INSGT_SL(nn.Module):
    '''
    wrapper for torch.istft to support batches
    Args:
         NSGT (Tensor): complex stft of
             shape (nb_samples, nb_channels, nb_bins, nb_frames, complex=2)
             last axis is stacked real and imaginary
        OR
             shape (nb_samples, nb_targets, nb_channels, nb_bins, nb_frames, complex=2)
             last axis is stacked real and imaginary
     '''
    def __init__(self, nsgt):
        super(INSGT_SL, self).__init__()
        self.nsgt = nsgt

    def _apply(self, fn):
        self.nsgt._apply(fn)
        return self

    def forward(self, X_list, length: int) -> Tensor:
        X_complex = [None]*len(X_list)
        for i, X in enumerate(X_list):
            Xshape = len(X.shape)

            X = torch.view_as_complex(X)

            shape = X.shape

            if Xshape == 6:
                X = X.view(X.shape[0]*X.shape[1], *X.shape[2:])
            else:
                X = X.view(X.shape[0]*X.shape[1]*X.shape[2], *X.shape[3:])

            # moveaxis back into into T x [packed-channels] x F1 x F2
            X = torch.moveaxis(X, -2, 0)

            X_complex[i] = X

        y = self.nsgt.nsgt.backward(X_complex, length)

        # unpack batch
        y = y.view(*shape[:-3], -1)

        return y


class ComplexNormSliCQT(nn.Module):
    r"""Compute the norm of complex tensor input.

    Extension of `torchaudio.functional.complex_norm` with mono

    Args:
        power (float): Power of the norm. (Default: `1.0`).
        mono (bool): Downmix to single channel after applying power norm
            to maximize
    """

    def __init__(self, power: float = 1.0, mono: bool = False):
        super(ComplexNormSliCQT, self).__init__()
        self.power = power
        self.mono = mono

    def forward(self, spec):
        # take the magnitude

        ret = [None]*len(spec)
        for i, C_block in enumerate(spec):
            C_block = torch.pow(torch.abs(torch.view_as_complex(C_block)), self.power)

            # downmix in the mag domain to preserve energy
            if self.mono:
                C_block = torch.mean(C_block, 1, keepdim=True)
            ret[i] = C_block

        return ret
