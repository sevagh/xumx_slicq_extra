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
from nsgt import NSGT, OctScale, MelScale, LogScale, BarkScale, VQLogScale

import json
from types import SimpleNamespace

from .ideal_mask import multichan_nsgt, multichan_insgt, TFTransform


mempool = cupy.get_default_memory_pool()


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
        description='describe tf configs'
    )
    parser.add_argument(
        'config_file',
        help='json file with time-frequency (stft, cqt) evaluation configs',
    )

    args = parser.parse_args()

    max_tracks = 1
    track_offset = 0

    # initiate musdb
    mus = musdb.DB(subsets='test', is_wav=True)

    # accumulate all time-frequency configs to compare
    tfs = []

    with open(args.config_file) as jf:
        config = json.load(jf)
        for track in mus.tracks[track_offset:max_tracks]:
            tmp = None
            for stft_win in config['stft_configs']['window_sizes']:
                tmp = {'type': 'stft', 'window': stft_win}
                tmp['name'] = f'stft-{stft_win}' # blank name for control config

                tf_transform = SimpleNamespace(**tmp)
                print(tmp)

                N = track.audio.shape[0]  # remember number of samples for future use
                tf = TFTransform(N, track.rate, tf_transform)

                describe_tf(track, tf)
                print()
                gc.collect()
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
                print(tmp)

                N = track.audio.shape[0]  # remember number of samples for future use
                tf = TFTransform(N, track.rate, tf_transform)

                describe_tf(track, tf)
                print()
                gc.collect()
