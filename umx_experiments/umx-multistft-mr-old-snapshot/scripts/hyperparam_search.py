import sys
import os
import musdb
import itertools
import librosa
import torch
import museval
import random
from functools import partial
import numpy as np
import argparse
from tqdm import tqdm

import scipy
from scipy.signal import stft, istft

import json
from types import SimpleNamespace

eps = 1.e-10


'''
from https://www.aicrowd.com/challenges/music-demixing-challenge-ismir-2021
    nb_sources, nb_samples, nb_channels = 4, 100000, 2
    references = np.random.rand(nb_sources, nb_samples, nb_channels)
    estimates = np.random.rand(nb_sources, nb_samples, nb_channels)
'''
def fast_sdr(track, estimates_dct, target):
    references = np.concatenate([np.expand_dims(source.audio, axis=0) for name, source in track.sources.items() if name == target])
    estimates = np.concatenate([np.expand_dims(est, axis=0) for est_name, est in estimates_dct.items() if est_name == target])

    # compute SDR for one song
    num = np.sum(np.square(references), axis=(1, 2)) + eps
    den = np.sum(np.square(references - estimates), axis=(1, 2)) + eps
    sdr_instr = 10.0 * np.log10(num / den)
    sdr_song = np.mean(sdr_instr)
    return np.median(sdr_song)


def phasemix_sep(X, Ymag):
    _, Xphase = librosa.magphase(X)

    return Ymag*Xphase


def ideal_mixphase(track, tf, plot=False):
    """
    ideal performance of magnitude from estimated source + phase of mix
    which is the default umx strategy for separation
    """
    N = track.audio.shape[0]

    X = tf.forward(track.audio)
    #print(f'{tf.name} X.shape: {X.shape}')

    #(I, F, T) = X.shape

    # Compute sources spectrograms
    P = {}
    # compute model as the sum of spectrograms
    model = eps

    # parallelize this
    for name, source in track.sources.items():
        # compute spectrogram of target source:
        # magnitude of STFT
        src_coef = tf.forward(source.audio)

        P[name] = np.abs(src_coef)

        # store the original, not magnitude, in the mix
        model += src_coef

    # now performs separation
    estimates = {}
    for name, source in track.sources.items():
        source_mag = P[name]

        #print('inverting phase')
        Yj = phasemix_sep(model, source_mag)

        # invert to time domain
        target_estimate = tf.backward(Yj, N)

        # set this as the source estimate
        estimates[name] = target_estimate

    return estimates


class TFTransform:
    def __init__(self, window=4096):
        self.nperseg = window
        self.noverlap = self.nperseg // 4

        self.name = f'stft-{self.nperseg}-{self.noverlap}'

    def forward(self, audio):
        return stft(audio.T, nperseg=self.nperseg, noverlap=self.noverlap)[-1].astype(np.complex64)

    def backward(self, X, len_x):
        return istft(X, nperseg=self.nperseg, noverlap=self.noverlap)[1].T.astype(np.float32)[:len_x]


class TrackEvaluator:
    def __init__(self, tracks):
        self.tracks = tracks

    def oracle(self, window=4096, printinfo=False):
        med_sdrs_bass = []
        med_sdrs_drums = []
        med_sdrs_vocals = []
        med_sdrs_other = []

        tf = TFTransform(window=window)

        if printinfo:
            print(f'{tf.name}')

        for track in tqdm(self.tracks, desc='tracks'):
            #track.chunk_start = 0
            #track.chunk_duration =5
            #print(f'track:\n\t{track.name}\n\t{track.chunk_duration}\n\t{track.chunk_start}')

            N = track.audio.shape[0]
            ests = ideal_mixphase(track, tf)

            med_sdrs_bass.append(fast_sdr(track, ests, target='bass'))
            med_sdrs_drums.append(fast_sdr(track, ests, target='drums'))
            med_sdrs_vocals.append(fast_sdr(track, ests, target='vocals'))
            med_sdrs_other.append(fast_sdr(track, ests, target='other'))

        # return 1 sdr per source
        return (
            np.mean(np.concatenate([np.expand_dims(med_sdr, axis=0) for med_sdr in med_sdrs_bass])),
            np.mean(np.concatenate([np.expand_dims(med_sdr, axis=0) for med_sdr in med_sdrs_drums])),
            np.mean(np.concatenate([np.expand_dims(med_sdr, axis=0) for med_sdr in med_sdrs_vocals])),
            np.mean(np.concatenate([np.expand_dims(med_sdr, axis=0) for med_sdr in med_sdrs_other])),
        )


def evaluate_single(f, params):
    curr_score_bass, curr_score_drums, curr_score_vocals, curr_score_other = f(window=params['windows'], printinfo=True)

    print('bass, drums, vocals, other sdr! {0:.2f} {1:.2f} {2:.2f} {3:.2f}'.format(
        curr_score_bass,
        curr_score_drums,
        curr_score_vocals,
        curr_score_other,
    ))
    print('total sdr: {0:.2f}'.format((curr_score_bass+curr_score_drums+curr_score_vocals+curr_score_other)/4))


def optimize_many(f, params, per_target, n_iter):
    if per_target:
        best_score_bass = float('-inf')
        best_param_bass = None

        best_score_drums = float('-inf')
        best_param_drums = None

        best_score_vocals = float('-inf')
        best_param_vocals = None

        best_score_other = float('-inf')
        best_param_other = None

        #print(f'optimizing target {target_name}')
        for _ in tqdm(range(n_iter)):
            window = random.choice(params['windows'])
            curr_score_bass, curr_score_drums, curr_score_vocals, curr_score_other = f(window=window)

            if curr_score_bass > best_score_bass:
                best_score_bass = curr_score_bass
                best_param_bass = window
                print('good bass sdr! {0}, {1}'.format(best_score_bass, best_param_bass))
            if curr_score_drums > best_score_drums:
                best_score_drums = curr_score_drums
                best_param_drums = window
                print('good drums sdr! {0}, {1}'.format(best_score_drums, best_param_drums))
            if curr_score_vocals > best_score_vocals:
                best_score_vocals = curr_score_vocals
                best_param_vocals = window
                print('good vocals sdr! {0}, {1}'.format(best_score_vocals, best_param_vocals))
            if curr_score_other > best_score_other:
                best_score_other = curr_score_other
                best_param_other = window
                print('good other sdr! {0}, {1}'.format(best_score_other, best_param_other))

        print(f'best scores')
        print(f'bass: \t{best_score_bass}\t{best_param_bass}')
        print(f'drums: \t{best_score_drums}\t{best_param_drums}')
        print(f'other: \t{best_score_other}\t{best_param_other}')
        print(f'vocals: \t{best_score_vocals}\t{best_param_vocals}')
    else:
        best_score_total = float('-inf')
        best_param_total = None

        for _ in tqdm(range(n_iter)):
            window = random.choice(params['windows'])
            curr_score_bass, curr_score_drums, curr_score_vocals, curr_score_other = f(window=window)
            tot = (curr_score_bass+curr_score_drums+curr_score_vocals+curr_score_other)/4

            if tot > best_score_total:
                best_score_total = tot
                best_param_total = window
                print('good total sdr! {0}, {1}'.format(best_score_total, best_param_total))
        print(f'best scores')
        print(f'total: \t{best_score_total}\t{best_param_total}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Search NSGT configs for best ideal mask'
    )
    parser.add_argument(
        '--windows',
        type=str,
        default='256,32768,128',
        help='comma-separated windows to evaluate (overlap is always //4)'
    )
    parser.add_argument(
        '--n-iter',
        type=int,
        default=60,
        help='number of iterations'
    )
    parser.add_argument(
        '--single',
        action='store_true',
        help='evaluate single nsgt instead of grid search'
    )
    parser.add_argument(
        '--per-target',
        action='store_true',
        help='maximize each target separately'
    )
    parser.add_argument(
        '--random-seed',
        type=int,
        default=42,
        help='rng seed'
    )

    args = parser.parse_args()

    random.seed(args.random_seed)

    # initiate musdb
    mus = musdb.DB(subsets='train', split='valid', is_wav=True)

    windows = (int(x) for x in args.windows.split(','))

    t = TrackEvaluator(mus.tracks)

    if not args.single:
        params = {
            'windows': list(np.arange(*windows)),
        }
        optimize_many(t.oracle, params, args.per_target, args.n_iter)
    else:
        params = {
            'windows': int(next(windows)),
        }
        evaluate_single(t.oracle, params)
