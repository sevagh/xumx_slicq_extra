import sys
import os
import musdb
import itertools
import museval
from museval.metrics import disable_cupy
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


class TrackEvaluator:
    def __init__(self, tracks, oracle='irm1'):
        self.tracks = tracks
        self.oracle_func = None
        if oracle == 'irm1':
            self.oracle_func = partial(ideal_mask, binary_mask=False, alpha=1)
        elif oracle == 'irm2':
            self.oracle_func = partial(ideal_mask, binary_mask=False, alpha=2)
        elif oracle == 'ibm1':
            self.oracle_func = partial(ideal_mask, binary_mask=True, alpha=1)
        elif oracle == 'ibm2':
            self.oracle_func = partial(ideal_mask, binary_mask=True, alpha=2)
        elif oracle == 'mpi':
            self.oracle_func = partial(ideal_mixphase, dur=5, start=20)
        elif oracle == 'fbin':
            self.oracle_func = partial(ideal_mask_fbin, dur=5, start=20)
        elif oracle == 'mbin':
            self.oracle_func = partial(ideal_mask_fbin, dur=5, start=20, mbin=True)
        elif oracle == 'svd':
            self.oracle_func = partial(slicq_svd, dur=5, start=20)
        elif oracle == 'global':
            self.oracle_func = partial(ideal_mask_mixphase_per_coef, dur=10, start=34.7)
        else:
            raise ValueError(f'unsupported oracle {oracle}')

    def eval_control(self, window_size=4096, eval_dir=None):
        all_bsses, single_score = self.oracle(control=True, stft_window=window_size, eval_dir=eval_dir)
        print(all_bsses)
        return single_score

    def eval_vqlog(self, fmin=20.0, fmax=22050, bins=12, gamma=25):
        return self.oracle(scale='vqlog', fmin=fmin, fmax=fmax, bins=bins, gamma=gamma, control=False)

    def eval_cqlog(self, fmin=20.0, fmax=22050, bins=12):
        return self.oracle(scale='cqlog', fmin=fmin, fmax=fmax, bins=bins, control=False)

    def eval_mel(self, fmin=20.0, fmax=22050, bins=12):
        return self.oracle(scale='mel', fmin=fmin, fmax=fmax, bins=bins, control=False)

    def eval_bark(self, fmin=20.0, fmax=22050, bins=12):
        return self.oracle(scale='bark', fmin=fmin, fmax=fmax, bins=bins, control=False)

    def oracle(self, scale='cqlog', fmin=20.0, fmax=22050, bins=12, gamma=25, control=False, stft_window=4096, eval_dir=None, transform_type_user=None):
        bins = int(bins)

        med_sdrs = []
        bss_scores_objs = []

        if transform_type_user is None:
            transform_type = "nsgt"
            if control:
                transform_type = "stft"
        else:
            transform_type = transform_type_user

        bss_scores_objs = []

        tf = TFTransform(44100, transform_type, stft_window, scale, fmin, fmax, bins, gamma)

        for track in tqdm(self.tracks):
            print(f'track: {track.name}')
            N = track.audio.shape[0]

            _, bss_scores_obj = self.oracle_func(track, tf, eval_dir=eval_dir, fast_eval=(not control))

            if control:
                bss_scores_objs.append(bss_scores_obj)
                bss_scores = bss_scores_obj.scores

                scores = np.zeros((4, 1), dtype=np.float32)
                for target_idx, t in enumerate(bss_scores['targets']):
                    if t['name'] == 'accompaniment':
                        continue
                    for metric_idx, metric in enumerate(['SDR']):
                        agg = np.nanmedian([np.float32(f['metrics'][metric]) for f in t['frames']])
                        scores[target_idx, metric_idx] = agg

                median_sdr = np.median(scores)
                med_sdrs.append(median_sdr)
            else:
                med_sdrs.append(bss_scores_obj)

        if control:
            tot = museval.EvalStore()
            [tot.add_track(t) for t in bss_scores_objs]

            return tot, np.median(med_sdrs)
        else:
            # fast_eval returns single sdr directly
            return np.median(med_sdrs)


def optimize(f, bounds, name, n_iter, n_random, logdir=None, randstate=1):
    #bounds_transformer = SequentialDomainReductionTransformer()

    optimizer = BayesianOptimization(
        f=f,
        pbounds=bounds,
        verbose=2,
        random_state=randstate,
        #bounds_transformer=bounds_transformer,
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
        '--control',
        action='store_true',
        help='evaluate control (stft)'
    )
    parser.add_argument(
        '--control-window-sizes',
        type=str,
        default='256,512,1024,1536,2048,3072,4096,8192,16384',
        help='comma-separated window sizes of stft to evaluate'
    )
    parser.add_argument(
        '--fixed-slicqt',
        action='store_true',
        help='evaluate fixed slicqt (no param search)'
    )
    parser.add_argument(
        '--fixed-slicqt-param',
        type=str,
        default='cqlog,20,20.0',
        help='comma-separated scale, bins, fmin, optionally gamma'
    )
    parser.add_argument(
        '--oracle',
        type=str,
        default='irm1',
        help='type of oracle to compute (choices: irm1, irm2, ibm1, ibm2, mpi)'
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
        '--cuda-device',
        type=int,
        default=0,
        help='which cuda device to run cupy bss eval on',
    )
    parser.add_argument(
        '--logdir',
        default=None,
        type=str,
        help='directory to store optimization logs',
    )
    parser.add_argument(
        '--eval-dir',
        default=None,
        type=str,
        help='directory to store evaluations',
    )
    parser.add_argument(
        '--split',
        type=str,
        default='valid',
        help='musdb data split'
    )

    args = parser.parse_args()
    print(args)

    cuda_dev = args.cuda_device % 2
    print(f'globally setting cuda device to: {cuda_dev}')
    cupy.cuda.runtime.setDevice(cuda_dev)

    random.seed(args.random_seed)
    disable_cupy()

    mus = None
    # initiate musdb
    if args.split == 'valid':
        mus = musdb.DB(subsets='train', split='valid', is_wav=True)
    elif args.split == 'test':
        mus = musdb.DB(subsets='test', is_wav=True)
    else:
        raise ValueError(f'musdb18 data split {args.split} unsupported')

    max_tracks = min(int(os.getenv('MUSDB_MAX_TRACKS', sys.maxsize)), len(mus.tracks))

    tracks = None
    if args.n_random_tracks:
        print(f'using {args.n_random_tracks} random tracks from MUSDB18-HQ train set validation split')
        tracks = random.sample(mus.tracks, args.n_random_tracks)
    else:
        print(f'using tracks 0-{max_tracks} from MUSDB18-HQ train set validation split')
        tracks = mus.tracks[:max_tracks]

    t = TrackEvaluator(tracks, oracle=args.oracle)

    print('oracle: {0}'.format(args.oracle))

    if args.control:
        for window_size in [int(x) for x in args.control_window_sizes.split(',')]:
            print(f'evaluating control stft {window_size} for oracle {args.oracle}')
            print('median SDR (no accompaniment): {0}'.format(
                t.eval_control(
                    window_size=window_size, eval_dir=os.path.join(args.eval_dir, f'{args.oracle}-{window_size}')
                )
            ))
        sys.exit(0)

    if args.fixed_slicqt:
        slicqt_args = args.fixed_slicqt_param.split(',')
        print(f'evaluating fixed slicqt {slicqt_args} for oracle {args.oracle}')
        scale, bins, fmin = slicqt_args
        fmin = float(fmin)
        bins = int(bins)
        print('median SDR (no accompaniment): {0}'.format(
            t.oracle(
                scale=scale,
                fmin=fmin,
                fmax=22050,
                bins=bins,
                control=True,
                eval_dir=os.path.join(args.eval_dir, f'{args.oracle}-{scale}-{bins}-{fmin:.1f}'),
                transform_type_user='nsgt',
            )
        ))
        sys.exit(0)

    if not args.control or args.fixed_slicqt:
        print('specify one of --control or --fixed-slicqt')
