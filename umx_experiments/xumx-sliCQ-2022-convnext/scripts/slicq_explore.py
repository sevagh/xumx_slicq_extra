import sys
import os
import gc
import musdb
import itertools
import torch
import museval
from functools import partial
import numpy as np
import random
import argparse
from xumx_slicq_22.transforms import make_filterbanks_slicqt, NSGTBase, phasemix_sep, ComplexNormSliCQT
from tqdm import tqdm

import scipy
from scipy.signal import stft, istft

import json
from types import SimpleNamespace

eps = 1.e-10


def _fast_sdr(track, estimates_dct, target, device):
    references = torch.cat([torch.unsqueeze(torch.tensor(source.audio.T, device=device), dim=0) for source_name, source in track.sources.items() if source_name == target])
    estimates = torch.cat([torch.unsqueeze(est, dim=0) for est_name, est in estimates_dct.items() if est_name == target])

    # compute SDR for one song
    num = torch.sum(torch.square(references), dim=(1, 2)) + eps
    den = torch.sum(torch.square(references - estimates), dim=(1, 2)) + eps
    sdr_target = 10.0 * torch.log10(num / den)
    return sdr_target


def ideal_mixphase(track, fwd, bwd, cnorm, device):
    """
    ideal performance of magnitude from estimated source + phase of mix
    which is the default umx strategy for separation
    """
    N = track.audio.shape[0]
    audio = torch.tensor(track.audio.T, device=device)

    # unsqueeze to add (1,) batch dimension
    X = fwd(torch.unsqueeze(audio, dim=0))

    # Compute sources spectrograms
    P = {}
    # compute model as the sum of spectrograms
    model = [eps]*len(X)

    # parallelize this
    for name, source in track.sources.items():
        # compute spectrogram of target source:
        # magnitude of STFT
        src_coef = fwd(torch.unsqueeze(torch.tensor(source.audio.T, device=device), dim=0))

        P[name] = cnorm(src_coef)

        # store the original, not magnitude, in the mix
        for i, src_coef_block in enumerate(src_coef):
            model[i] += src_coef_block + eps

    # now performs separation
    estimates = {}
    for name, source in track.sources.items():
        source_mag = P[name]

        #Yj = [None]*len(model)
        #for i, model_block in enumerate(model):
        Yj = phasemix_sep(model, source_mag)

        # invert to time domain
        target_estimate = bwd(Yj, N)

        # set this as the source estimate
        estimates[name] = torch.squeeze(target_estimate, dim=0)

    return estimates


class TrackEvaluator:
    def __init__(self, tracks, max_sllen, device="cuda"):
        self.tracks = tracks
        self.max_sllen = max_sllen
        self.device = device

    def oracle(self, scale='cqlog', fmin=20.0, bins=12, gamma=25):
        bins = int(bins)

        med_sdrs_bass = []
        med_sdrs_drums = []
        med_sdrs_vocals = []
        med_sdrs_other = []

        n = NSGTBase(scale=scale, fbins=bins, fmin=fmin, device=self.device, gamma=gamma)

        # skip too big transforms
        if n.sllen > self.max_sllen:
            return (
                float('-inf'),
                float('-inf'),
                float('-inf'),
                float('-inf'),
                n.sllen
            )
        print(f'sllen, trlen: {n.sllen}, {n.trlen}')

        nsgt, insgt = make_filterbanks_slicqt(n)

        cnorm = ComplexNormSliCQT().to(self.device)

        for track in tqdm(self.tracks):
            #print(f'track:\n\t{track.name}\n\t{track.chunk_duration}\n\t{track.chunk_start}')

            N = track.audio.shape[0]
            ests = ideal_mixphase(track, nsgt.forward, insgt.forward, cnorm.forward, device=self.device)

            med_sdrs_bass.append(_fast_sdr(track, ests, target='bass', device=self.device))
            med_sdrs_drums.append(_fast_sdr(track, ests, target='drums', device=self.device))
            med_sdrs_vocals.append(_fast_sdr(track, ests, target='vocals', device=self.device))
            med_sdrs_other.append(_fast_sdr(track, ests, target='other', device=self.device))

            del ests
            torch.cuda.empty_cache()
            gc.collect()

        # return 1 sdr per source
        return (
            torch.mean(torch.cat([torch.unsqueeze(med_sdr, dim=0) for med_sdr in med_sdrs_bass])),
            torch.mean(torch.cat([torch.unsqueeze(med_sdr, dim=0) for med_sdr in med_sdrs_drums])),
            torch.mean(torch.cat([torch.unsqueeze(med_sdr, dim=0) for med_sdr in med_sdrs_vocals])),
            torch.mean(torch.cat([torch.unsqueeze(med_sdr, dim=0) for med_sdr in med_sdrs_other])),
            n.sllen
        )


def evaluate_single(f, params):
    curr_score_bass, curr_score_drums, curr_score_vocals, curr_score_other, sllen = f(scale=params['scale'], fmin=params['fmin'], bins=params['bins'], gamma=params['gamma'])

    print('bass, drums, vocals, other sdr! {0:.2f} {1:.2f} {2:.2f} {3:.2f}'.format(
        curr_score_bass,
        curr_score_drums,
        curr_score_vocals,
        curr_score_other,
    ))
    print('total sdr: {0:.2f}'.format((curr_score_bass+curr_score_drums+curr_score_vocals+curr_score_other)/4))


def optimize_many(f, params, n_iter, per_target):
    if per_target:
        best_score_bass = float('-inf')
        best_param_bass = None

        best_score_drums = float('-inf')
        best_param_drums = None

        best_score_vocals = float('-inf')
        best_param_vocals = None

        best_score_other = float('-inf')
        best_param_other = None

        fmins = list(np.arange(*params['fmin']))
        gammas = list(np.arange(*params['gamma']))

        #print(f'optimizing target {target_name}')
        for _ in tqdm(range(n_iter)):
            while True: # loop in case we skip for exceeding sllen
                scale = random.choice(params['scales'])
                bins = np.random.randint(*params['bins'])
                fmin = random.choice(fmins)
                gamma = random.choice(gammas)
                
                curr_score_bass, curr_score_drums, curr_score_vocals, curr_score_other, sllen = f(scale=scale, fmin=fmin, bins=bins, gamma=gamma)

                params_tup = (scale, bins, fmin, gamma, sllen)

                if curr_score_bass == curr_score_drums and curr_score_drums == curr_score_vocals and curr_score_vocals == curr_score_other and curr_score_other == float('-inf'):
                    # sllen not supported
                    print('reroll for sllen...')
                    continue

                if curr_score_bass > best_score_bass:
                    best_score_bass = curr_score_bass
                    best_param_bass = params_tup
                    print('good bass sdr! {0}, {1}'.format(best_score_bass, best_param_bass))
                if curr_score_drums > best_score_drums:
                    best_score_drums = curr_score_drums
                    best_param_drums = params_tup
                    print('good drums sdr! {0}, {1}'.format(best_score_drums, best_param_drums))
                if curr_score_vocals > best_score_vocals:
                    best_score_vocals = curr_score_vocals
                    best_param_vocals = params_tup
                    print('good vocals sdr! {0}, {1}'.format(best_score_vocals, best_param_vocals))
                if curr_score_other > best_score_other:
                    best_score_other = curr_score_other
                    best_param_other = params_tup
                    print('good other sdr! {0}, {1}'.format(best_score_other, best_param_other))
                break
        print(f'best scores')
        print(f'bass: \t{best_score_bass}\t{best_param_bass}')
        print(f'drums: \t{best_score_drums}\t{best_param_drums}')
        print(f'other: \t{best_score_other}\t{best_param_other}')
        print(f'vocals: \t{best_score_vocals}\t{best_param_vocals}')
    else:
        best_score_total = float('-inf')
        best_param_total = None

        fmins = list(np.arange(*params['fmin']))
        gammas = list(np.arange(*params['gamma']))

        for _ in tqdm(range(n_iter)):
            while True: # loop in case we skip for exceeding sllen
                scale = random.choice(params['scales'])
                bins = np.random.randint(*params['bins'])
                fmin = random.choice(fmins)
                gamma = random.choice(gammas)
                
                curr_score_bass, curr_score_drums, curr_score_vocals, curr_score_other, sllen = f(scale=scale, fmin=fmin, bins=bins, gamma=gamma)
                tot = (curr_score_bass+curr_score_drums+curr_score_vocals+curr_score_other)/4
                # hack to maximize negative score
                #tot *= -1

                params_tup = (scale, bins, fmin, gamma, sllen)

                if curr_score_bass == curr_score_drums and curr_score_drums == curr_score_vocals and curr_score_vocals == curr_score_other and curr_score_other == float('-inf'):
                    # sllen not supported
                    print('reroll for sllen...')
                    continue

                if tot > best_score_total:
                    best_score_total = tot
                    best_param_total = params_tup
                    print('good total sdr! {0}, {1}'.format(best_score_total, best_param_total))
                break
        print(f'best scores')
        print(f'total: \t{best_score_total}\t{best_param_total}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Search sliCQT configs for best ideal mask'
    )
    parser.add_argument("--root", type=str, help="root path of dataset")
    parser.add_argument(
        '--bins',
        type=str,
        default='10,300',
        help='comma-separated range of bins to evaluate'
    )
    parser.add_argument(
        '--fmins',
        type=str,
        default='10,130,0.1',
        help='comma-separated range of fmin to evaluate'
    )
    parser.add_argument(
        '--gammas',
        type=str,
        default='0,100,0.1',
        help='comma-separated range of gamma to evaluate'
    )
    parser.add_argument(
        '--n-iter',
        type=int,
        default=60,
        help='number of iterations'
    )
    parser.add_argument(
        '--fscale',
        type=str,
        default='bark,mel,cqlog,vqlog',
        help='nsgt frequency scales, csv (choices: vqlog, cqlog, mel, bark)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        help='torch device (cpu vs cuda)'
    )
    parser.add_argument(
        "--cuda-device", type=int, default=-1, help="choose which gpu to train on (-1 = 'cuda' in pytorch)"
    )
    parser.add_argument(
        '--random-seed',
        type=int,
        default=42,
        help='rng seed to pick the same random 5 songs'
    )
    parser.add_argument(
        '--max-sllen',
        type=int,
        default=32768,
        help='maximum sllen above which to skip iterations'
    )
    parser.add_argument(
        '--single',
        action='store_true',
        help='evaluate single nsgt instead of randomized param search'
    )
    parser.add_argument(
        '--per-target',
        action='store_true',
        help='maximize each target separately'
    )

    args = parser.parse_args()

    random.seed(args.random_seed)

    if args.cuda_device >= 0:
        print(f'setting device to {torch.cuda.get_device_name(args.cuda_device)}')
        device = torch.device(args.cuda_device)
    else:
        device = args.device

    # initiate musdb
    mus = musdb.DB(root=args.root, subsets='train', split='valid', is_wav=True)

    if not args.single:
        scales = args.fscale.split(',')
        bins = tuple([int(x) for x in args.bins.split(',')])
        fmins = tuple([float(x) for x in args.fmins.split(',')])
        gammas = tuple([float(x) for x in args.gammas.split(',')])
        print(f'Parameter ranges to evaluate:\n\tscales: {scales}\n\tbins: {bins}\n\tfmins: {fmins}\n\tgammas: {gammas}')
        print(f'Ignoring fscales that exceed sllen {args.max_sllen}')

        params = {
            'scales': scales,
            'bins': bins,
            'fmin': fmins,
            'gamma': gammas,
        }

        t = TrackEvaluator(mus.tracks, args.max_sllen, device=device)
        optimize_many(t.oracle, params, args.n_iter, args.per_target)
    else:
        params = {
            'scale': args.fscale,
            'bins': int(args.bins),
            'fmin': float(args.fmins),
            'gamma': float(args.gammas),
        }

        print(f'Parameter to evaluate:\n\t{params}')

        t = TrackEvaluator(mus.tracks, args.max_sllen, device=device)
        evaluate_single(t.oracle, params)
