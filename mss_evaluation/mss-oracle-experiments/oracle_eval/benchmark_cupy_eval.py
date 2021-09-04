import sys
import os
import musdb
import itertools
import museval
from museval.metrics import clear_cupy_cache, disable_cupy
import cupy
from functools import partial
import numpy as np
import random
import argparse

from shared import TFTransform 
from oracle import ideal_mask_fbin, ideal_mask, ideal_mixphase, slicq_svd, ideal_mask_mixphase_per_coef

from tqdm import tqdm
import scipy
from scipy.signal import stft, istft

import json
from types import SimpleNamespace
import time


def fake_rand_est(track):
    N = track.audio.shape[0]

    # now performs separation
    estimates = {}
    accompaniment_source = 0
    for name, source in track.sources.items():
        target_estimate = np.random.randn(*source.audio.shape)

        # random source estimate
        estimates[name] = target_estimate

        # accumulate to the accompaniment if this is not vocals
        if name != 'vocals':
            accompaniment_source += target_estimate

    estimates['accompaniment'] = accompaniment_source

    bss_scores = museval.eval_mus_track(
        track,
        estimates,
    )

    return estimates, bss_scores


class TrackEvaluator:
    def __init__(self, tracks, bench_iter, disable_cupy):
        self.tracks = tracks
        self.bench_iters = bench_iter
        self.disable_cupy = disable_cupy

    def bench(self):
        tot = 0.
        pbar_bench = tqdm(range(self.bench_iters))
        pbar_bench.set_description("Benchmark iteration")
        for _ in pbar_bench:
            start = time.time()
            pbar_tracks = tqdm(self.tracks)
            pbar_tracks.set_description("Track")
            for track in pbar_tracks:
                print('\nbss eval...\n')
                fake_rand_est(track)

                if not self.disable_cupy:
                    print('\nclearing cupy cache...\n')
                    clear_cupy_cache()
            tot += (time.time() - start)

        tot /= float(self.bench_iters)
        return tot


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Benchmark cupy metrics eval'
    )
    parser.add_argument(
        '--bench-iter',
        type=int,
        default=100,
        help='benchmark iterations',
    )
    parser.add_argument(
        '--random-seed',
        type=int,
        default=42,
        help='random seed for RNG',
    )
    parser.add_argument(
        '--split',
        type=str,
        default='valid',
        help='musdb data split'
    )
    parser.add_argument(
        '--cuda-device',
        type=int,
        default=0,
        help='cuda device'
    )
    parser.add_argument(
        '--disable-cupy',
        action="store_true",
        help='disable cupy in bss eval'
    )

    args = parser.parse_args()
    print(args)

    cuda_dev = args.cuda_device % 2
    print(f'globally setting cuda device to: {cuda_dev}')
    cupy.cuda.runtime.setDevice(cuda_dev)

    if args.disable_cupy:
        print('disabling cupy in bss eval')
        disable_cupy()

    random.seed(args.random_seed)
    np.random.seed(args.random_seed)

    mus = None
    # initiate musdb
    if args.split == 'valid':
        mus = musdb.DB(subsets='train', split='valid', is_wav=True)
    elif args.split == 'test':
        mus = musdb.DB(subsets='test', is_wav=True)
    else:
        raise ValueError(f'musdb18 data split {args.split} unsupported')

    max_tracks = min(int(os.getenv('MUSDB_MAX_TRACKS', sys.maxsize)), len(mus.tracks))

    print(f'using tracks 0-{max_tracks} from MUSDB18-HQ {args.split} set')
    tracks = mus.tracks[:max_tracks]

    t = TrackEvaluator(tracks, args.bench_iter, args.disable_cupy)

    time_elapsed = t.bench() 
    print(f'time elapsed: {time_elapsed}')
    sys.exit(0)
