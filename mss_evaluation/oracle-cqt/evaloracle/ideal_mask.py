import sys
import os
import musdb
import gc
import itertools
import museval
import numpy as np
import functools
import argparse
import cupy
from cupy.fft.config import get_plan_cache
import tqdm

import scipy
from scipy.signal import stft, istft

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

        self.nsgt = None
        self.tf_transform = tf_transform
        self.N = ntrack
        self.nsgt = None
        if use_nsgt:
            scl = None
            if tf_transform.scale == 'octave':
                scl = OctScale(tf_transform.fmin, fs/2, tf_transform.bins)
            elif tf_transform.scale == 'mel':
                scl = MelScale(tf_transform.fmin, fs/2, tf_transform.bins)
            else:
                raise ValueError("only 'octave' supported for now")

            # nsgt has a multichannel=True param which blows memory up. prefer to do it myself
            self.nsgt = NSGT(scl, fs, self.N, real=True, matrixform=True)

    def forward(self, audio):
        if not self.nsgt:
            return stft(audio.T, nperseg=self.tf_transform.window)[-1].astype(np.complex64)
        else:
            return stereo_nsgt(audio, self.nsgt)

    def backward(self, X):
        if not self.nsgt:
            return istft(X)[1].T[:self.N, :].astype(np.float32)
        else:
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

        # multiply the mix by the mask
        Yj = np.multiply(X, Mask)

        # invert to time domain
        target_estimate = tf.backward(Yj)

        # set this as the source estimate
        estimates[name] = target_estimate

        # accumulate to the accompaniment if this is not vocals
        if name != 'vocals':
            accompaniment_source += target_estimate

    estimates['accompaniment'] = accompaniment_source

    gc.collect()
    mempool = cupy.get_default_memory_pool()

    # cupy disable fft caching to free blocks
    fft_cache = get_plan_cache()
    fft_cache.set_size(0)

    mempool.free_all_blocks()

    # cupy reenable fft caching
    fft_cache.set_size(16)
    fft_cache.set_memsize(-1)

    if eval_dir is not None:
        museval.eval_mus_track(
            track,
            estimates,
            output_dir=eval_dir,
        )

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

    masks = [
            {'power': 1, 'binary': False},
            {'power': 2, 'binary': False},
            {'power': 1, 'binary': True},
            {'power': 2, 'binary': True},
    ]

    mss_evaluations = list(itertools.product(mus.tracks[:max_tracks], tfs, masks))

    for (track, tf_transform, mask) in tqdm.tqdm(mss_evaluations):
        N = track.audio.shape[0]  # remember number of samples for future use
        tf = TFTransform(N, track.rate, tf_transform)

        # construct mask name e.g. irm1, ibm2
        mask_name = 'i'
        if mask['binary']:
            mask_name += 'b'
        else:
            mask_name += 'r'
        mask_name += f"m{str(mask['power'])}"

        est = ideal_mask(
            track,
            tf,
            mask['power'],
            mask['binary'],
            0.5,
            os.path.join(args.eval_dir, f'{mask_name}-{tf_transform.name}'))

        gc.collect()

        if args.audio_dir:
            mus.save_estimates(est, track, os.path.join(args.eval_dir, f'{mask_name}-{tf_transform.name}'))
