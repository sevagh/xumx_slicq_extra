import sys
import os
import musdb
import itertools
import numpy as np
import random
import argparse
from shared import TFTransform, dimensionality_cmp


class TrackEvaluator:
    def __init__(self, tracks):
        self.tracks = tracks

    def dimcmp(self, tfs):
        for track in self.tracks:
            print(f'{track=}')
            return dimensionality_cmp(track, tfs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Compare NSGT and STFT dimensionality'
    )
    parser.add_argument(
        '--window',
        type=int,
        default=4096,
        help='stft window size',
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

    args = parser.parse_args()

    random.seed(args.random_seed)

    # initiate musdb
    mus = musdb.DB(subsets='train', split='valid', is_wav=True)

    max_tracks = min(int(os.getenv('MUSDB_MAX_TRACKS', sys.maxsize)), len(mus.tracks))

    tracks = None
    if args.n_random_tracks:
        print(f'using {args.n_random_tracks} random tracks from MUSDB18-HQ train set validation split')
        tracks = random.sample(mus.tracks, args.n_random_tracks)
    else:
        print(f'using tracks 0-{max_tracks} from MUSDB18-HQ train set validation split')
        tracks = mus.tracks[:max_tracks]

    # stft
    tf_stft_1 = TFTransform(44100, window=1024)
    tf_stft_2 = TFTransform(44100, window=4096)
    tf_stft_3 = TFTransform(44100, window=8192)

    t = TrackEvaluator(tracks)

    most_negative_nyquist_delta = 0

    while True:
        fmin = random.uniform(10.0, 130.0)
        fmax = random.uniform(10000.0, 22050.0)
        fbins = random.randrange(10, 300)

        print(f'looking for degenerate nsgts with {fmin=} {fmax=} {fbins=}...')

        print(f'comparing {fmax} with 22050')
        tf_1 = TFTransform(44100, transform_type="nsgt", fscale="cqlog", fbins=fbins, fmin=fmin, fmax=fmax)
        tf_2 = TFTransform(44100, transform_type="nsgt", fscale="cqlog", fbins=fbins, fmin=fmin, fmax=22050)

        totels = t.dimcmp([tf_stft_1, tf_stft_2, tf_stft_3, tf_1, tf_2])

        nyquist_delta = totels[-1]-totels[0]
        print(f'nyquist_delta: {nyquist_delta}')
        print(f'most_negative_nyquist_delta: {most_negative_nyquist_delta}')

        # nyquist is smaller
        if nyquist_delta < most_negative_nyquist_delta:
            most_negative_nyquist_delta = nyquist_delta

            print(f'\nNEW BIGGER DEGENERATE CASE! {fbins} {fmin} {fmax}\n')
