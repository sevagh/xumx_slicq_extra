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


import scipy
from scipy.signal import stft, istft

# use CQT based on nonstationary gabor transform
from nsgt import NSGT, OctScale, MelScale, LogScale, VQLogScale, BarkScale

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


class TrackEvaluator:
    def __init__(self, tracks, rate, min_N):
        self.tracks = tracks
        self.min_N = min_N
        self.rate = rate

    def eval_control(self):
        return self.ideal_mask(alpha=1, binary_mask=False, control=True)

    def eval_vqlog(self, fmin=20.0, bins=12, gamma=25):
        return self.ideal_mask(scale='vqlog', fmin=fmin, bins=bins, gamma=gamma, alpha=1, binary_mask=False, control=False)

    def eval_cqlog(self, fmin=20.0, bins=12):
        return self.ideal_mask(scale='cqlog', fmin=fmin, bins=bins, alpha=1, binary_mask=False, control=False)

    def eval_mel(self, fmin=20.0, bins=12):
        return self.ideal_mask(scale='mel', fmin=fmin, bins=bins, alpha=1, binary_mask=False, control=False)

    def eval_bark(self, fmin=20.0, bins=12):
        return self.ideal_mask(scale='bark', fmin=fmin, bins=bins, alpha=1, binary_mask=False, control=False)

    def ideal_mask(self, scale='cqlog', fmin=20.0, bins=12, gamma=25, alpha=2, binary_mask=False, theta=0.5, control=False):
        bins = int(bins)
        med_sdrs = []

        N = self.min_N

        scl = None
        if scale == 'mel':
            scl = MelScale(fmin, 22050, bins)
        elif scale == 'bark':
            scl = BarkScale(fmin, 22050, bins)
        elif scale == 'cqlog':
            scl = LogScale(fmin, 22050, bins)
        elif scale == 'vqlog':
            scl = VQLogScale(fmin, 22050, bins, gamma=gamma)
        else:
            raise ValueError(f"unsupported scale {scale}")

        nsgt = NSGT(scl, self.rate, N, real=True, matrixform=True)

        for track in self.tracks:
            # even nsgt size to save some time
            track.chunk_start = 0
            track.chunk_duration = round(self.min_N/track.rate)

            # small epsilon to avoid dividing by zero
            eps = np.finfo(np.float32).eps

            if control:
                X = stft(track.audio.T, nperseg=2048)[-1].astype(np.complex64)
            else:
                X = multichan_nsgt(track.audio, nsgt)

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
                    if control:
                        P[name] = np.abs(stft(source.audio.T, nperseg=2048)[-1].astype(np.complex64))**alpha
                    else:
                        P[name] = np.abs(multichan_nsgt(source.audio, nsgt))**alpha
                    model += P[name]

            # now performs separation
            estimates = {}
            accompaniment_source = 0
            for name, source in track.sources.items():
                if binary_mask:
                    # compute STFT of target source
                    if control:
                        Yj = stft(source.audio.T, nperseg=2048)[-1].astype(np.complex64)
                    else:
                        Yj = multichan_nsgt(source.audio, nsgt)

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
                if control:
                    target_estimate = istft(Yj)[1].T[:N, :].astype(np.float32)
                else:
                    target_estimate = multichan_insgt(Yj, nsgt)

                # set this as the source estimate
                estimates[name] = target_estimate

                # accumulate to the accompaniment if this is not vocals
                if name != 'vocals':
                    accompaniment_source += target_estimate

            estimates['accompaniment'] = accompaniment_source

            gc.collect()

            # cupy disable fft caching to free blocks
            fft_cache = get_plan_cache()
            fft_cache.set_size(0)

            mempool.free_all_blocks()

            # cupy reenable fft caching
            fft_cache.set_size(16)
            fft_cache.set_memsize(-1)

            bss_scores = museval.eval_mus_track(
                track,
                estimates,
            ).scores

            scores = np.zeros((4, 1), dtype=np.float32)
            for target_idx, t in enumerate(bss_scores['targets']):
                if t['name'] == 'accompaniment':
                    continue
                for metric_idx, metric in enumerate(['SDR']):
                    agg = np.nanmedian([np.float32(f['metrics'][metric]) for f in t['frames']])
                    scores[target_idx, metric_idx] = agg

            med_sdrs.append(np.median(scores))

        return np.median(med_sdrs)


def optimize(f, bounds, logdir, name, n_iter, n_random):
    logpath = os.path.join(args.logdir, f"./{name}_logs.json")

    optimizer = BayesianOptimization(
        f=f,
        pbounds=bounds,
        verbose=2,
        random_state=1,
    )
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
        'logdir',
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

    min_N = min([track.audio.shape[0] for track in tracks])
    t = TrackEvaluator(tracks, tracks[0].rate, 8843231)

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

    optimize(t.eval_vqlog, pbounds_vqlog, args.logdir, "vqlog", args.optimization_iter, args.optimization_random)
    optimize(t.eval_cqlog, pbounds_other, args.logdir, "cqlog", args.optimization_iter, args.optimization_random)
    optimize(t.eval_mel, pbounds_other, args.logdir, "mel", args.optimization_iter, args.optimization_random)
    optimize(t.eval_bark, pbounds_other, args.logdir, "bark", args.optimization_iter, args.optimization_random)
