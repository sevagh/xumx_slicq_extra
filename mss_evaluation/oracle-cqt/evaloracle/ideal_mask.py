import sys
import os
import musdb
import gc
import itertools
import museval
import numpy as np
import functools
import argparse
from timeit import default_timer as timer
import cupy
from cupy.fft.config import get_plan_cache
import tqdm

import scipy
from scipy.signal import stft, istft
#from memory_profiler import profile

# use CQT based on nonstationary gabor transform
from nsgt import NSGT, OctScale, MelScale, LogScale # BarkScale

import json
from types import SimpleNamespace


def stereo_nsgt(audio, nsgt):
    X_chan1 = np.asarray(nsgt.forward(audio[:, 0]))
    X_chan2 = np.asarray(nsgt.forward(audio[:, 1]))
    X = np.empty((2, X_chan1.shape[0], X_chan1.shape[1]), dtype=np.complex64)
    X[0, :, :] = X_chan1
    X[1, :, :] = X_chan2
    return X


def stereo_insgt(C, nsgt):
    C_chan1 = C[0, :, :]
    C_chan2 = C[1, :, :]
    inv1 = nsgt.backward(C_chan1)
    inv2 = nsgt.backward(C_chan2)
    ret_audio = np.empty((2, inv1.shape[0]), dtype=np.float32)
    ret_audio[0, :] = inv1
    ret_audio[1, :] = inv2
    return ret_audio.T


class TFTransform:
    def __init__(self, ntrack, fs, tf_transform):
        use_nsgt = (tf_transform.type == "nsgt")
        print(f'using TF spectrogram with params: {tf_transform}')

        self.nsgt = None
        self.tf_transform = tf_transform
        self.N = ntrack
        self.nsgt = None
        if use_nsgt:
            scl = None
            if tf_transform.scale == 'octave':
                # use nyquist as maximum frequency always
                scl = OctScale(tf_transform.fmin, fs/2, tf_transform.bins)
                # nsgt has a multichannel=True param which blows memory up. prefer to do it myself
            else:
                raise ValueError("only 'octave' supported for now")

            self.nsgt = NSGT(scl, fs, self.N, real=True, matrixform=True)

    def forward(self, audio):
        if not self.nsgt:
            print('stft')
            return stft(audio.T, nperseg=self.tf_transform.window)[-1].astype(np.complex64)
        else:
            print('nsgt')
            return stereo_nsgt(audio, self.nsgt)

    def backward(self, X):
        if not self.nsgt:
            print('istft')
            return istft(X)[1].T[:self.N, :].astype(np.float32)
        else:
            print('insgt')
            return stereo_insgt(X, self.nsgt)


def ideal_mask(track, tf, alpha=2, binary_mask=False, theta=0.5, eval_dir=None):
    """
    if theta=None:
        Ideal Ratio Mask:
        processing all channels inpependently with the ideal ratio mask.
        this is the ratio of spectrograms, where alpha is the exponent to take for
        spectrograms. usual values are 1 (magnitude) and 2 (power)

    if theta=float:
        Ideal Binary Mask:
        processing all channels inpependently with the ideal binary mask.

        the mix is send to some source if the spectrogram of that source over that
        of the mix is greater than theta, when the spectrograms are take as
        magnitude of STFT raised to the power alpha. Typical parameters involve a
        ratio of magnitudes (alpha=1) and a majority vote (theta = 0.5)
    """

    # small epsilon to avoid dividing by zero
    eps = np.finfo(np.float32).eps

    print('evaluating track {0}'.format(track))

    print('1. forward transform')
    X = tf.forward(track.audio)

    (I, F, T) = X.shape

    # soft mask stuff
    if not binary_mask:
        # Compute sources spectrograms
        P = {}
        # compute model as the sum of spectrograms
        model = eps

        # parallelize this
        for name, source in track.sources.items():
            # compute spectrogram of target source:
            # magnitude of STFT to the power alpha
            print('2. forward transform for item {0}'.format(name))
            P[name] = np.abs(tf.forward(source.audio))**alpha
            model += P[name]

    # now performs separation
    estimates = {}
    accompaniment_source = 0
    for name, source in track.sources.items():
        if binary_mask:
            # compute STFT of target source
            Yj = tf.forward(source.audio)

            # Create Binary Mask
            Mask = np.divide(np.abs(Yj)**alpha, (eps + np.abs(X)**alpha))
            Mask[np.where(Mask >= theta)] = 1
            Mask[np.where(Mask < theta)] = 0
        else:
            # compute soft mask as the ratio between source spectrogram and total
            Mask = np.divide(np.abs(P[name]), model)

        print('3. apply mask {0}'.format(name))

        # multiply the mix by the mask
        print(f'\tmask and transform dim: {Mask.shape}, {X.shape}')
        Yj = np.multiply(X, Mask)

        print('4. inverse transform {0}'.format(name))

        # invert to time domain
        target_estimate = tf.backward(Yj)

        # set this as the source estimate
        estimates[name] = target_estimate

        # accumulate to the accompaniment if this is not vocals
        if name != 'vocals':
            accompaniment_source += target_estimate

    estimates['accompaniment'] = accompaniment_source
    for ename, e_val in estimates.items():
        print(f'{ename} {e_val.dtype}')

    gc.collect()
    mempool = cupy.get_default_memory_pool()
    print('mempool pre-free: {0}'.format(mempool.used_bytes()))

    # cupy disable fft caching to free blocks
    fft_cache = get_plan_cache()
    fft_cache.set_size(0)

    mempool.free_all_blocks()
    print('mempool post-free: {0}'.format(mempool.used_bytes()))

    # cupy reenable fft caching
    fft_cache.set_size(16)
    fft_cache.set_memsize(-1)

    print('5. BSS eval step')
    start = timer()
    if eval_dir is not None:
        museval.eval_mus_track(
            track,
            estimates,
            output_dir=eval_dir,
        )
    end = timer()
    print(f'6. done {end-start}')

    return estimates


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Evaluate Ideal Ratio Mask'
    )
    parser.add_argument(
        '--audio_dir',
        nargs='?',
        help='Folder where audio results are saved',
        default=None,
    )

    parser.add_argument(
        '--eval_dir',
        nargs='?',
        help='Folder where evaluation results are saved'
    )
    parser.add_argument(
        'config_file',
        help='json file with time-frequency (stft, cqt) evaluation configs',
    )

    args = parser.parse_args()

    max_tracks = int(os.getenv('MUSDB_MAX_TRACKS', sys.maxsize))

    # initiate musdb
    mus = musdb.DB(subsets='test', is_wav=True)

    # accumulate all time-frequency configs to compare
    tfs = []

    with open(args.config_file) as jf:
        configs = json.load(jf)
        for config in configs:
            tf_transform = SimpleNamespace(**config)
            tfs.append(tf_transform)

    mss_evaluations = itertools.product(mus.tracks[:max_tracks], tfs)

    masks = {
            'irm1': (1, False),
            'irm2': (2, False),
            'ibm1': (1, True),
            'ibm2': (2, True),
    }

    for (track, tf_transform) in tqdm.tqdm(mss_evaluations):
        N = track.audio.shape[0]  # remember number of samples for future use
        tf = TFTransform(N, track.rate, tf_transform)

        for mask_name, mask_params in masks.items():
            est = ideal_mask(
                track,
                tf,
                mask_params[0],
                mask_params[1],
                0.5,
                os.path.join(args.eval_dir, f'{mask_name}-{tf_transform.name}'))

            gc.collect()

            if args.audio_dir:
                mus.save_estimates(est, track, os.path.join(args.eval_dir, f'{mask_name}-{tf_transform.name}'))
