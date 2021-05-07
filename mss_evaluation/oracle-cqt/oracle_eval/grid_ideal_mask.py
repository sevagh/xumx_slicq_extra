import sys
import os
import musdb
import gc
import itertools
import museval
import numpy as np
import random
import librosa
import functools
import argparse
import pandas as pd
from io import StringIO
import cupy
from cupy.fft.config import get_plan_cache
import tqdm
from collections import defaultdict
from bayes_opt import BayesianOptimization
from bayes_opt.util import load_logs
from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events

from shared import ideal_mask, ideal_mixphase, TFTransform


import scipy
from scipy.signal import stft, istft

# use CQT based on nonstationary gabor transform
from nsgt import NSGT, OctScale, MelScale, LogScale, VQLogScale, BarkScale

import json
from types import SimpleNamespace


class TrackEvaluator:
    def __init__(self, tracks, phasemix=False):
        self.tracks = tracks
        self.phasemix = phasemix # switch between IRM1 and phasemix

    def eval_control(self, window_size=4096):
        return self.ideal_mask(alpha=1, binary_mask=False, control=True, stft_window=window_size)

    def eval_vqlog(self, fmin=20.0, bins=12, gamma=25):
        return self.ideal_mask(scale='vqlog', fmin=fmin, bins=bins, gamma=gamma, alpha=1, binary_mask=False, control=False)

    def eval_cqlog(self, fmin=20.0, bins=12):
        return self.ideal_mask(scale='cqlog', fmin=fmin, bins=bins, alpha=1, binary_mask=False, control=False)

    def eval_mel(self, fmin=20.0, bins=12):
        return self.ideal_mask(scale='mel', fmin=fmin, bins=bins, alpha=1, binary_mask=False, control=False)

    def eval_bark(self, fmin=20.0, bins=12):
        return self.ideal_mask(scale='bark', fmin=fmin, bins=bins, alpha=1, binary_mask=False, control=False)

    def ideal_mask(self, scale='cqlog', fmin=20.0, bins=12, gamma=25, alpha=2, binary_mask=False, theta=0.5, control=False, stft_window=4096):
        bins = int(bins)

        med_sdrs = []

        transform_type = "nsgt"
        if control:
            transform_type = "stft"

        for track in self.tracks:
            #print(f'track: {track.name}')
            N = track.audio.shape[0]

            tf = TFTransform(N, track.rate, transform_type, stft_window, scale, fmin, bins, gamma)

            bss_scores = None
            if self.phasemix:
                _, bss_scores = ideal_mixphase(track, tf, eval_dir=None)
            else:
                _, bss_scores = ideal_mask(track, tf, alpha=alpha, binary_mask=binary_mask, theta=theta, eval_dir=None)

            scores = np.zeros((4, 1), dtype=np.float32)
            for target_idx, t in enumerate(bss_scores['targets']):
                if t['name'] == 'accompaniment':
                    continue
                for metric_idx, metric in enumerate(['SDR']):
                    agg = np.nanmedian([np.float32(f['metrics'][metric]) for f in t['frames']])
                    scores[target_idx, metric_idx] = agg

            median_sdr = np.median(scores)
            med_sdrs.append(median_sdr)

        return np.median(med_sdrs)


def optimize(f, bounds, name, n_iter, n_random, logdir=None):
    optimizer = BayesianOptimization(
        f=f,
        pbounds=bounds,
        verbose=2,
        random_state=1,
    )
    if logdir:
        logpath = os.path.join(logdir, f"./{name}_logs.json")
        try:
            load_logs(optimizer, logs=[logpath])
            print('loaded previous log')
        except FileNotFoundError:
            print('no log found, re-optimizing')
            pass

        logger = JSONLogger(path=logpath)
        optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)

    print(f'optimizing {name} scale')
    optimizer.maximize(
        init_points=n_random,
        n_iter=n_iter,
    )
    print(f'max {name}: {optimizer.max}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Search NSGT configs for best ideal mask'
    )
    parser.add_argument(
        '--mono',
        action='store_true',
        help='use mono channel (faster evaluation)'
    )
    parser.add_argument(
        '--control',
        action='store_true',
        help='evaluate control (stft)'
    )
    parser.add_argument(
        '--phasemix',
        action='store_true',
        help='phasemix'
    )
    parser.add_argument(
        '--n-random-tracks',
        type=int,
        default=None,
        help='use N random tracks instead of MUSDB_MAX_TRACKS'
    )
    parser.add_argument(
        '--random-seed',
        type=int,
        default=42,
        help='rng seed to pick the same random 5 songs'
    )
    parser.add_argument(
        '--optimization-iter',
        type=int,
        default=2,
        help='bayesian optimization iterations',
    )
    parser.add_argument(
        '--optimization-random',
        type=int,
        default=1,
        help='bayesian optimization random iterations',
    )
    parser.add_argument(
        '--logdir',
        default=None,
        type=str,
        help='directory to store optimization logs',
    )

    args = parser.parse_args()

    random.seed(args.random_seed)

    # initiate musdb
    mus = musdb.DB(subsets='test', is_wav=True, mono=args.mono)

    max_tracks = min(int(os.getenv('MUSDB_MAX_TRACKS', sys.maxsize)), len(mus.tracks))

    tracks = None
    if args.n_random_tracks:
        print(f'using {args.n_random_tracks} random tracks from MUSDB18-HQ test set')
        tracks = random.sample(mus.tracks, args.n_random_tracks)
    else:
        print(f'using tracks 0-{max_tracks} from MUSDB18-HQ test set')
        tracks = mus.tracks[:max_tracks]

    t = TrackEvaluator(tracks, phasemix=args.phasemix)

    bins = (12,348)
    fmins = (15.0,60.0)
    gammas = (0.0,100.0)

    pbounds_vqlog = {
        'bins': bins,
        'fmin': fmins,
        'gamma': gammas,
    }

    pbounds_other = {
        'bins': bins,
        'fmin': fmins
    }

    if args.control:
        print('evaluating control stft')
        print('window size 4096: {0}'.format(t.eval_control(window_size=4096)))
        print('window size 1024: {0}'.format(t.eval_control(window_size=1024)))
        print('window size 16384: {0}'.format(t.eval_control(window_size=16384)))
        sys.exit(0)

    optimize(t.eval_vqlog, pbounds_vqlog, "vqlog", args.optimization_iter, args.optimization_random, logdir=args.logdir)
    optimize(t.eval_cqlog, pbounds_other, "cqlog", args.optimization_iter, args.optimization_random, logdir=args.logdir)
    optimize(t.eval_mel, pbounds_other, "mel", args.optimization_iter, args.optimization_random, logdir=args.logdir)
    optimize(t.eval_bark, pbounds_other, "bark", args.optimization_iter, args.optimization_random, logdir=args.logdir)
