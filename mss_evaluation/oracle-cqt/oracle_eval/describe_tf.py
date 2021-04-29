import sys
import os
import musdb
import librosa
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
from nsgt import NSGT, OctScale, MelScale, LogScale, BarkScale

import json
from types import SimpleNamespace


mempool = cupy.get_default_memory_pool()


def multichan_nsgt(audio, nsgt):
    n_chan = audio.shape[1]
    Xs = []
    for i in range(n_chan):
        Xs.append(np.asarray(nsgt.forward(audio[:, i])))
    return np.asarray(Xs).astype(np.complex64)


def multichan_insgt(C, nsgt):
    n_chan = C.shape[0]
    rets = []
    for i in range(n_chan):
        C_chan = C[i, :, :]
        inv = nsgt.backward(C_chan)
        rets.append(inv)
    ret_audio = np.asarray(rets)
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
            if tf_transform.scale == 'oct':
                scl = OctScale(tf_transform.fmin, tf_transform.fmax, tf_transform.bins)
            elif tf_transform.scale == 'mel':
                scl = MelScale(tf_transform.fmin, tf_transform.fmax, tf_transform.bins)
            elif tf_transform.scale == 'bark':
                scl = BarkScale(tf_transform.fmin, tf_transform.fmax, tf_transform.bins)
            elif tf_transform.scale == 'log':
                scl = LogScale(tf_transform.fmin, tf_transform.fmax, tf_transform.bins)
            else:
                raise ValueError(f"unsupported scale {tf_transform.scale}")

            print(f'\tscale: {tf_transform.scale}')
            print(f'\tfmin: {tf_transform.fmin}')
            print('\tconstant-Q: {0:.2f} Hz'.format(scl.Q()))
            print(f'\tbins: {tf_transform.bins}')
            pitches, _ = scl()
            print(f'\ttotal frequencies: {len(pitches)}')
            print(f'\tpitches')
            for octave_idx in range(12, len(pitches), 12):
                octave_pitches = pitches[octave_idx-12:octave_idx]

                print('\t\t')
                [print('{0:.2f},{1}\t'.format(p, librosa.hz_to_note(p)), end='') for p in octave_pitches]
            print()

            # nsgt has a multichannel=True param which blows memory up. prefer to do it myself
            self.nsgt = NSGT(scl, fs, self.N, real=True, matrixform=True)

    def forward(self, audio):
        if not self.nsgt:
            return stft(audio.T, nperseg=self.tf_transform.window)[-1].astype(np.complex64)
        else:
            return multichan_nsgt(audio, self.nsgt)

    def backward(self, X):
        if not self.nsgt:
            return istft(X)[1].T[:self.N, :].astype(np.float32)
        else:
            return multichan_insgt(X, self.nsgt)


def describe_tf(track, tf):
    # small epsilon to avoid dividing by zero
    eps = np.finfo(np.float32).eps

    print(f'track dtype, dimensions: {track.audio.dtype}, {track.audio.shape}')
    print('track duration: {0:.2f} min'.format(len(track.audio)/track.rate/60.0))
    X = tf.forward(track.audio)
    print(f'tf transform dtype, dimensions, size: {X.dtype}, {X.shape}, {X.size}')

    gc.collect()

    # cupy disable fft caching to free blocks
    fft_cache = get_plan_cache()
    fft_cache.set_size(0)

    mempool.free_all_blocks()

    # cupy reenable fft caching
    fft_cache.set_size(16)
    fft_cache.set_memsize(-1)

    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Evaluate Ideal Ratio Mask'
    )
    parser.add_argument(
        'config_file',
        help='json file with time-frequency (stft, cqt) evaluation configs',
    )

    args = parser.parse_args()

    max_tracks = 2
    track_offset = 0

    # initiate musdb
    mus = musdb.DB(subsets='test', is_wav=True)

    # accumulate all time-frequency configs to compare
    tfs = []

    with open(args.config_file) as jf:
        config = json.load(jf)
        tmp = None
        for stft_win in config['stft_configs']['window_sizes']:
            tmp = {'type': 'stft', 'window': stft_win}
            tmp['name'] = '' # blank name for control config

            tf_transform = SimpleNamespace(**tmp)
            tfs.append(tf_transform)
        for nsgt_conf in itertools.product(
                config['nsgt_configs']['scale'],
                config['nsgt_configs']['fmin'],
                config['nsgt_configs']['fmax'],
                config['nsgt_configs']['bins'],
                ):
            tmp = {'type': 'nsgt', 'scale': nsgt_conf[0], 'fmin': nsgt_conf[1], 'fmax': nsgt_conf[2], 'bins': nsgt_conf[3]}
            fmin_str = f'{nsgt_conf[1]}'
            fmin_str = fmin_str.replace('.', '')
            tmp['name'] = f'{nsgt_conf[0]}-{fmin_str}-{nsgt_conf[3]}'

            tf_transform = SimpleNamespace(**tmp)

            for track in mus.tracks[track_offset:max_tracks]:
                print(tmp)

                N = track.audio.shape[0]  # remember number of samples for future use
                tf = TFTransform(N, track.rate, tf_transform)

                describe_tf(track, tf)
                gc.collect()
