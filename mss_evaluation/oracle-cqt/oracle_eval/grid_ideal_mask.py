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
from collections import defaultdict

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



def ideal_mask(track, scale='log', fmin='20.0', bins='12', alpha=2, binary_mask=False, theta=0.5):
    N = track.audio.shape[0]

    scl = None
    if scale == 'mel':
        scl = MelScale(fmin, 22050, bins)
    elif scale == 'bark':
        scl = BarkScale(fmin, 22050, bins)
    elif scale == 'log':
        scl = LogScale(fmin, 22050, bins)
    else:
        raise ValueError(f"unsupported scale {scale}")

    nsgt = NSGT(scl, track.rate, N, real=True, matrixform=True)

    # small epsilon to avoid dividing by zero
    eps = np.finfo(np.float32).eps

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
            P[name] = np.abs(multichan_nsgt(source.audio, nsgt))**alpha
            model += P[name]

    # now performs separation
    estimates = {}
    accompaniment_source = 0
    for name, source in track.sources.items():
        if binary_mask:
            # compute STFT of target source
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

    return scores


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
        '--print-every-n',
        type=int,
        default=10,
        help='print leaderboard this often'
    )
    parser.add_argument(
        '--chunk-duration',
        type=float,
        default=10.0,
        help='duration of each chunk in seconds'
    )

    args = parser.parse_args()

    max_tracks = int(os.getenv('MUSDB_MAX_TRACKS', sys.maxsize))
    track_offset = int(os.getenv('MUSDB_TRACK_OFFSET', 0))

    # initiate musdb
    mus = musdb.DB(subsets='test', is_wav=True, mono=args.mono)

    # exhaustive search over nsgt parameters
    scales = ['log', 'mel', 'bark']

    # start with jumps of 12
    bins = list(np.arange(12, 193, 12))

    # start with jumps of 10
    fmins = list(np.arange(20,101,10))

    scores = defaultdict(list)
    n_iter = 0

    curr_chunk = 0

    for track in itertools.cycle(mus.tracks[track_offset:max_tracks]):
        print(f'evaluating track {track.name}, chunk {curr_chunk}-{curr_chunk+args.chunk_duration} s')
        track.chunk_duration = args.chunk_duration
        track.chunk_start = curr_chunk

        # do 1 eval run on first chunk_duration of each track 
        # then for the next chunk_duration, etc.
        # over time we slowly converge toward having evaluated every single track

        for (scale, fmin, bin_) in itertools.product(scales, fmins, bins):
            print(f'evaluating nsgt {scale} {fmin} {bin_}')
            # use IRM1, ideal ratio mask with magnitude spectrogram (1 = |S|^1)
            score = ideal_mask(track, scale=scale, fmin=fmin, bins=bin_, alpha=1, binary_mask=False)
            scores[(scale, bin_, fmin)].append(score)

            n_iter += 1

            if n_iter % 10 == 0:
                print('top 5 tf configs so far, median score (all targets x metrics)')

                for (tf_conf, scrs) in sorted(scores.items(), key=lambda item: np.median(item[1]), reverse=True)[:5]:
                    # take median across all chunks
                    med_scrs = np.median(scrs, axis=0)
                    print(f'\t{tf_conf}:\n{print_scores_ndarray(med_scrs)}')

        # move chunk_duration up
        curr_chunk += args.chunk_duration
