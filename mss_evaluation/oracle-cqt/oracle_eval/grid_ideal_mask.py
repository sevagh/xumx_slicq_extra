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


def print_scores_ndarray(scores):
    # reverse of this:
    #scores = np.zeros((5, 4), dtype=np.float32)
    #for target_idx, t in enumerate(bss_scores['targets']):
    #    for metric_idx, metric in enumerate(['SDR', 'SIR', 'ISR', 'SAR']):
    #        agg = np.nanmedian([np.float32(f['metrics'][metric]) for f in t['frames']])
    #        scores[target_idx, metric_idx] = agg
    targets = ['drums', 'bass', 'other', 'vocals', 'accompaniment']
    metrics = ['SDR', 'SIR', 'ISR', 'SAR']

    out = ""
    for target_idx in np.ndindex(scores.shape[:1]):
        target = targets[target_idx[0]]
        out += target.ljust(16) + "==> "
        for metric_idx in np.ndindex(scores.shape[1:]):
            metric = metrics[metric_idx[0]]
            out += metric + ":" + "{:>8.3f}".format(scores[target_idx, metric_idx][0]) + "  "
        out += "\n"
    return out


def print_scores_csv(scores):
    targets = ['drums', 'bass', 'other', 'vocals', 'accompaniment']
    metrics = ['SDR', 'SIR', 'ISR', 'SAR']

    out = ""
    for idx, score in np.ndenumerate(scores):
        target = targets[idx[0]]
        metric = targets[idx[1]]
        out += "{:.3f},".format(score)
    return out


def ideal_mask(track, scale='cqlog', fmin='20.0', bins=12, gamma=25, alpha=2, binary_mask=False, theta=0.5, control=False):
    N = track.audio.shape[0]

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

    #if not control:
    #    pitches, Q = scl()
    #    print(f'\ttotal frequencies: {len(pitches)}')
    #    print(f'\tpitches')
    #    for octave_idx in range(12, max(len(pitches), 24), 12):
    #        octave_pitches = pitches[octave_idx-12:octave_idx]
    #        print('\t\t')
    #        [print('{0:.2f},{1}\t'.format(p, librosa.hz_to_note(p)), end='') for p in octave_pitches]
    #    print(f'{Q=}')

    nsgt = NSGT(scl, track.rate, N, real=True, matrixform=True)

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

    scores = np.zeros((5, 4), dtype=np.float32)
    for target_idx, t in enumerate(bss_scores['targets']):
        for metric_idx, metric in enumerate(['SDR', 'SIR', 'ISR', 'SAR']):
            agg = np.nanmedian([np.float32(f['metrics'][metric]) for f in t['frames']])
            scores[target_idx, metric_idx] = agg

    return scores, X


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Grid search over NSGT for Ideal Mask source separation'
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

    #random.seed(42)
    args = parser.parse_args()

    max_tracks = int(os.getenv('MUSDB_MAX_TRACKS', sys.maxsize))

    # initiate musdb
    mus = musdb.DB(subsets='test', is_wav=True, mono=args.mono)
    tracks = None
    if args.n_random_tracks:
        print(f'using {args.n_random_tracks} random tracks')
        tracks = random.sample(mus.tracks, args.n_random_tracks)
    else:
        print(f'using tracks 0-MUSDB_MAX_TRACKS')
        tracks = mus.tracks[:max_tracks]

    # exhaustive search over all nsgt parameters
    scales = ['vqlog', 'cqlog', 'mel', 'bark']

    bins = list(np.arange(12, 333, 12))

    fmins = list(np.arange(15.0, 55.0, 5.0))

    gammas = list(np.arange(0.0, 100.0, 5.0))

    scores = defaultdict(list)
    coef_count = defaultdict(list)
    n_iter = 0

    targets = ['drums', 'bass', 'other', 'vocals', 'accompaniment']
    metrics = ['SDR', 'SIR', 'ISR', 'SAR']
    header = 'tf_config,'
    for p in itertools.product(targets, metrics):
        header += '.'.join(p) + ','
    header += 'bss_per_coef,coef_size'

    for track in tracks:
        print(f'evaluating track {track.name}')

        print(f'first evaluating control, stft 2048')
        # use IRM1, ideal ratio mask with magnitude spectrogram (1 = |S|^1) - control, stft 2048
        score, transform = ideal_mask(track, scale='mel', fmin=20, bins=12, alpha=1, binary_mask=False, control=True)
        tf = ('stft', '2048')
        scores[tf].append(score)
        coef_count[tf].append(transform.size)

        for (scale, fmin, bin_) in itertools.product(scales, fmins, bins):
            if scale == 'vqlog':
                # iterate over variable-q gamma factors separately
                for gamma in gammas:
                    print(f'evaluating variable-q nsgt {scale} {fmin} {bin_} {gamma}')
                    # use IRM1, ideal ratio mask with magnitude spectrogram (1 = |S|^1)
                    score, transform = ideal_mask(track, scale=scale, fmin=fmin, bins=bin_, gamma=gamma, alpha=1, binary_mask=False, control=False)
                    scores[(scale, bin_, fmin, gamma)].append(score)
                    coef_count[(scale, bin_, fmin, gamma)].append(transform.size)
                    n_iter += 1
            else:
                print(f'evaluating nsgt {scale} {fmin} {bin_}')
                # use IRM1, ideal ratio mask with magnitude spectrogram (1 = |S|^1)
                score, transform = ideal_mask(track, scale=scale, fmin=fmin, bins=bin_, gamma=gamma, alpha=1, binary_mask=False, control=False)
                scores[(scale, bin_, fmin, gamma)].append(score)
                coef_count[(scale, bin_, fmin, gamma)].append(transform.size)
                n_iter += 1

            # every 10 iterations, print leaderboard
            if n_iter % 10 == 0:
                print('top 5 tf configs so far, median score/coefficient count (all targets x metrics)')

                for (tf_conf, scrs) in sorted(scores.items(), key=lambda item: np.median(item[1]), reverse=True)[:5]:
                    # take median across all tracks
                    med_scrs = np.median(scores[tf_conf], axis=0) # take median across all tracks
                    print(f'\t{tf_conf}:\n{print_scores_ndarray(med_scrs)}')

    print('\nCSV: all tf configs, sorted by desc median score/coef count (all targets x metrics)\n')

    all_csv = header
    for (tf_conf, scrs) in sorted(scores.items(), key=lambda item: np.median(item[1]), reverse=True):
        med_scrs = np.median(scores[tf_conf], axis=0) # take median across all tracks

        med_coef_size = np.median(coef_count[tf_conf])
        med_scr_per_coef = np.median(scrs)/med_coef_size # take median across all tracks

        med_coef_size_int = int(med_coef_size)

        tf_conf_name = '-'.join([str(t) for t in tf_conf])
        all_csv += f"\n{tf_conf_name},{print_scores_csv(med_scrs)[:-1]},{med_coef_size_int},{med_scr_per_coef}"

    df = pd.read_csv(StringIO(all_csv), index_col=0)

    print(all_csv)
