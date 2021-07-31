import sys
import gc
import os
import musdb
import itertools
import torch
from collections import defaultdict
import museval
from functools import partial
import numpy as np
import random
import argparse
from openunmix.transforms import make_filterbanks, NSGTBase, phasemix_sep, ComplexNorm
from bayes_opt import BayesianOptimization
from bayes_opt.util import load_logs
from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events
from tqdm import tqdm

import scipy
from scipy.signal import stft, istft

eps = 1.e-10


def _fast_sdr(track, estimates_dct, device):
    references = torch.cat([torch.unsqueeze(torch.tensor(source.audio.T, device=device), dim=0) for source in track.sources.values()])
    estimates = torch.cat([torch.unsqueeze(est, dim=0) for est_name, est in estimates_dct.items() if est_name != 'accompaniment'])

    # compute SDR for one song
    num = torch.sum(torch.square(references), dim=(1, 2)) + eps
    den = torch.sum(torch.square(references - estimates), dim=(1, 2)) + eps
    sdr_target = 10.0 * torch.log10(num / den)
    sdr_song = torch.mean(sdr_target)
    return sdr_target


def stft_fwd(audio):
    return torch.stft(audio, n_fft=4096, hop_length=1024, return_complex=True).type(torch.complex64)


def stft_bwd(X, N):
    return torch.istft(X, n_fft=4096, hop_length=1024, length=N)


def ideal_mixphase_stft(track, device):
    """
    ideal performance of magnitude from estimated source + phase of mix
    which is the default umx strategy for separation
    """
    N = track.audio.shape[0]
    audio = torch.tensor(track.audio.T, device=device)

    # unsqueeze to add (1,) batch dimension
    X = stft_fwd(audio)

    # Compute sources spectrograms
    P = {}
    # compute model as the sum of spectrograms
    model = eps

    # parallelize this
    for name, source in track.sources.items():
        # compute spectrogram of target source:
        # magnitude of STFT
        src_coef = torch.view_as_real(stft_fwd(torch.tensor(source.audio.T, device=device)))
 
        P[name] = torch.abs(torch.view_as_complex(src_coef))

        # store the original, not magnitude, in the mix
        model += src_coef

    # now performs separation
    estimates = {}
    for name, source in track.sources.items():
        source_mag = P[name]

        Yj = phasemix_sep(model, source_mag)

        # invert to time domain
        target_estimate = stft_bwd(torch.view_as_complex(Yj), N)

        # set this as the source estimate
        estimates[name] = target_estimate

    return estimates


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
    model = defaultdict(lambda: eps)

    # parallelize this
    for name, source in track.sources.items():
        # compute spectrogram of target source:
        # magnitude of STFT
        src_coef = fwd(torch.unsqueeze(torch.tensor(source.audio.T, device=device), dim=0))

        P[name] = cnorm(src_coef)

        # store the original, not magnitude, in the mix
        for time_bucket, src_coef_block in src_coef.items():
            model[time_bucket] += src_coef_block

    # now performs separation
    estimates = {}
    for name, source in track.sources.items():
        source_mag = P[name]

        Yj = {}
        for time_bucket, model_block in model.items():
            Yj[time_bucket] = phasemix_sep(model_block, source_mag[time_bucket])

        # invert to time domain
        target_estimate = bwd(Yj, N)

        # set this as the source estimate
        estimates[name] = torch.squeeze(target_estimate, dim=0)

    return estimates


class TrackEvaluator:
    def __init__(self, tracks, max_sllen, scale, device="cuda", offset=0):
        self.tracks = tracks
        self.max_sllen = max_sllen
        self.device = device
        self.scale = scale
        self.cnorm = ComplexNorm().to(device)
        self.offset = offset

    def oracle(self, fmin=20.0, bins=12, gamma=25):
        bins = int(np.floor(bins))

        # set degenerate conditions/frequency scales
        # 1. sllen too high (> max sllen)
        # 2. unordered fbins/jagged
        try:
            n = NSGTBase(self.scale, bins, fmin, device=self.device)
        except ValueError as e:
            return float('-inf')

        # TODO: reroll logic here
        nsgt, insgt = make_filterbanks(n)

        med_sdrs = []

        for track in tqdm(self.tracks):
            N = track.audio.shape[0]
            #print(f'evaluating {track}, {(N/44100):.2f} seconds long')
            ests = ideal_mixphase(track, nsgt.forward, insgt.forward, self.cnorm.forward, device=self.device)
            med_sdrs.append(_fast_sdr(track, ests, self.device))

            gc.collect()

        # return 1 total sdr
        return torch.mean(torch.cat([torch.unsqueeze(med_sdr, dim=0) for med_sdr in med_sdrs])) + self.offset


def optimize_many(f, name, bounds, n_iter, n_random_iter, logdir=None, randstate=42):
    optimizer = BayesianOptimization(
        f=f,
        pbounds=bounds,
        verbose=2,
        random_state=randstate,
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
        init_points=n_random_iter,
        n_iter=n_iter,
    )
    print(f'max {name}: {optimizer.max}')


if __name__ == '__main__':
    mus = musdb.DB(subsets='train', split='valid', is_wav=True)

    control_sdrs = []
    nsgt_sdrs = []

    scale = 'mel'
    bins = 515
    fmin = 45.0
    gamma = 1.7

    # set degenerate conditions/frequency scales
    # 1. sllen too high (> max sllen)
    # 2. unordered fbins/jagged
    device = "cuda"

    n = NSGTBase(scale, bins, fmin, sllen=None, device=device, gamma=gamma)

    print(f'sllen: {n.sllen}')

    # TODO: reroll logic here
    nsgt, insgt = make_filterbanks(n)

    med_sdrs = []

    cnorm = ComplexNorm().to(device)

    for track in tqdm(mus.tracks):
        N = track.audio.shape[0]
        #print(f'evaluating {track}, {(N/44100):.2f} seconds long')
        ests_stft = ideal_mixphase_stft(track, device=device)

        control_sdrs.append(_fast_sdr(track, ests_stft, device))

        ests_nsgt = ideal_mixphase(track, nsgt.forward, insgt.forward, cnorm.forward, device=device)

        nsgt_sdrs.append(_fast_sdr(track, ests_nsgt, device))

        gc.collect()
        del ests_stft
        del ests_nsgt
        torch.cuda.empty_cache()

    control_sdr = torch.mean(torch.cat([torch.unsqueeze(control_sdr, dim=0) for control_sdr in control_sdrs]))
    nsgt_sdr = torch.mean(torch.cat([torch.unsqueeze(nsgt_sdr, dim=0) for nsgt_sdr in nsgt_sdrs]))
    print(f'Control score: {control_sdr:.2f}')
    print(f'NSGT score: {nsgt_sdr:.2f}')


#if __name__ == '__main__':
#    parser = argparse.ArgumentParser(
#        description='Search NSGT configs for best ideal mask'
#    )
#    parser.add_argument(
#        '--bins',
#        type=str,
#        default='10,2000',
#        help='comma-separated range of bins to evaluate'
#    )
#    parser.add_argument(
#        '--fmins',
#        type=str,
#        default='10,130',
#        help='comma-separated range of fmin to evaluate'
#    )
#    parser.add_argument(
#        '--gammas',
#        type=str,
#        default='0,100',
#        help='comma-separated range of gamma to evaluate'
#    )
#    parser.add_argument(
#        '--n-iter',
#        type=int,
#        default=800,
#        help='number of iterations'
#    )
#    parser.add_argument(
#        '--n-random-iter',
#        type=int,
#        default=200,
#        help='number of initial random iterations'
#    )
#    parser.add_argument(
#        '--fscale',
#        type=str,
#        default='bark',
#        help='nsgt frequency scale (default: bark)'
#    )
#    parser.add_argument(
#        '--random-seed',
#        type=int,
#        default=42,
#        help='rng seed for Bayesian optimization process'
#    )
#    parser.add_argument(
#        '--max-sllen',
#        type=int,
#        default=int(6*44100),
#        help='maximum sllen above which to skip iterations'
#    )
#    parser.add_argument(
#        '--logdir',
#        default=None,
#        type=str,
#        help='directory to store optimization logs',
#    )
#    parser.add_argument(
#        "--no-cuda", action="store_true", default=False, help="disables CUDA training"
#    )
#    parser.add_argument(
#        "--cuda-device", type=int, default=-1, help="choose which gpu to train on (-1 = 'cuda' in pytorch)"
#    )
#
#    args = parser.parse_args()
#
#    use_cuda = not args.no_cuda and torch.cuda.is_available()
#    device = torch.device("cuda" if use_cuda else "cpu")
#
#    if use_cuda and args.cuda_device >= 0:
#        device = torch.device(args.cuda_device)
#
#    random.seed(args.random_seed)
#
#    # initiate musdb
#    mus = musdb.DB(subsets='train', split='valid', is_wav=True)
#    bins = tuple([int(x) for x in args.bins.split(',')])
#    fmins = tuple([float(x) for x in args.fmins.split(',')])
#    gammas = tuple([float(x) for x in args.gammas.split(',')])
#
#    pbounds_vqlog = {
#        'bins': bins,
#        'fmin': fmins,
#        'gamma': gammas,
#    }
#
#    pbounds_other = {
#        'bins': bins,
#        'fmin': fmins,
#    }
#
#    control_sdrs = []
#
#    for track in tqdm(mus.tracks):
#        N = track.audio.shape[0]
#        #print(f'evaluating {track}, {(N/44100):.2f} seconds long')
#        ests = ideal_mixphase_stft(track, device=device)
#        control_sdrs.append(_fast_sdr(track, ests, device))
#
#    control_sdr = torch.mean(torch.cat([torch.unsqueeze(control_sdr, dim=0) for control_sdr in control_sdrs]))
#    print(f'Control score to beat: {control_sdr:.2f}')
#
#    t = TrackEvaluator(mus.tracks, args.max_sllen, args.fscale, device=device, offset=-control_sdr)
#
#    if args.fscale == 'vqlog':
#        optimize_many(t.oracle, args.fscale, pbounds_vqlog, args.n_iter, args.n_random_iter, logdir=args.logdir, randstate=args.random_seed)
#    else:
#        optimize_many(t.oracle, args.fscale, pbounds_other, args.n_iter, args.n_random_iter, logdir=args.logdir, randstate=args.random_seed)
