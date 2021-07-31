import sys
import os
import musdb
import itertools
import torch
import museval
from functools import partial
import numpy as np
import random
import argparse
from nsgt import NSGT_sliced, MelScale, LogScale, BarkScale, VQLogScale
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
def fast_sdr(track, estimates_dct, device):
    references = torch.cat([torch.unsqueeze(torch.tensor(source.audio, device=device), dim=0) for name, source in track.sources.items()])
    estimates = torch.cat([torch.unsqueeze(est, dim=0) for est_name, est in estimates_dct.items() if est_name != 'accompaniment'])

    # compute SDR for one song
    num = torch.sum(torch.square(references), dim=(1, 2)) + eps
    den = torch.sum(torch.square(references - estimates), dim=(1, 2)) + eps
    sdr_instr = 10.0 * torch.log10(num / den)
    sdr_song = torch.mean(sdr_instr)
    return torch.median(sdr_song)


def atan2(y, x):
    r"""Element-wise arctangent function of y/x.
    Returns a new tensor with signed angles in radians.
    It is an alternative implementation of torch.atan2

    Args:
        y (Tensor): First input tensor
        x (Tensor): Second input tensor [shape=y.shape]

    Returns:
        Tensor: [shape=y.shape].
    """
    pi = 2 * torch.asin(torch.tensor(1.0))
    x += ((x == 0) & (y == 0)) * 1.0
    out = torch.atan(y / x)
    out += ((y >= 0) & (x < 0)) * pi
    out -= ((y < 0) & (x < 0)) * pi
    out *= 1 - ((y > 0) & (x == 0)) * 1.0
    out += ((y > 0) & (x == 0)) * (pi / 2)
    out *= 1 - ((y < 0) & (x == 0)) * 1.0
    out += ((y < 0) & (x == 0)) * (-pi / 2)
    return out


def phasemix_sep(X, Ymag):
    Xphase = atan2(X[..., 1], X[..., 0])
    Ycomplex = torch.empty_like(X)

    Ycomplex[..., 0] = Ymag * torch.cos(Xphase)
    Ycomplex[..., 1] = Ymag * torch.sin(Xphase)
    return Ycomplex


def assemble_coefs(cqt, ncoefs):
    """
    Build a sequence of blocks out of incoming overlapping CQT slices
    """
    cqt = iter(cqt)
    cqt0 = next(cqt)
    cq0 = np.asarray(cqt0).T
    shh = cq0.shape[0]//2
    print('shh')
    out = np.empty((ncoefs, cq0.shape[1], cq0.shape[2]), dtype=cq0.dtype)
    
    fr = 0
    sh = max(0, min(shh, ncoefs-fr))

    print('fr: {0}'.format(fr))
    print('sh: {0}'.format(sh))
    print('shh: {0}'.format(shh))

    out[fr:fr+sh] = cq0[sh:] # store second half

    # add up slices
    for i, cqi in enumerate(cqt):
        print('on slice {0}'.format(i))
        cqi = np.asarray(cqi).T
        print('cqi shape: {0}'.format(cqi.shape))
        out[fr:fr+sh] += cqi[:sh]
        cqi = cqi[sh:]
        fr += sh
        sh = max(0, min(shh, ncoefs-fr))
        out[fr:fr+sh] = cqi[:sh]
        
    return out[:fr]


def ideal_mixphase(track, tf, plot=False):
    """
    ideal performance of magnitude from estimated source + phase of mix
    which is the default umx strategy for separation
    """
    N = track.audio.shape[0]

    print(f'song duration: {N/tf.fs} seconds')

    X = tf.forward(track.audio)

    time_seq_steps = X.shape[-1]
    print(f'{tf.name} X.shape: {X.shape}')

    if plot:
        print("Plotting t*f space")
        import matplotlib.pyplot as plt

        # magnitudes are fine here
        X = torch.abs(X)

        # mono for plotting
        print(f'X.shape: {X.shape}')
        X = X.mean(1, keepdim=False)
        print(f'X.shape: {X.shape}')

        mp_kern = 100
        mp = torch.nn.MaxPool1d(mp_kern, mp_kern, return_indices=True)

        X_temporal_pooled, inds = mp(X)
        print(f'X_temporal_pooled.shape: {X_temporal_pooled.shape}')

        mup = torch.nn.MaxUnpool1d(mp_kern, mp_kern)

        X_recon = mup(X_temporal_pooled, inds, output_size=X.size())
        print(f'X_recon.shape: {X_recon.shape}')

        norm = lambda x: torch.sqrt(torch.sum(torch.abs(torch.square(torch.abs(x)))))
        rec_err = norm(X_recon-X)/norm(X)

        print(f'recon after maxpool: {rec_err}')
        X_temporal_pooled = X_temporal_pooled.reshape(X_temporal_pooled.shape[0]*X_temporal_pooled.shape[-1], X_temporal_pooled.shape[-2])

        X_1 = X.reshape(X.shape[0]*X.shape[-1], -1).detach().cpu().numpy()
        X_2 = X_temporal_pooled.detach().cpu().numpy()
        X_3 = X_recon.reshape(X_recon.shape[0]*X_recon.shape[-1], -1).detach().cpu().numpy()

        print(f'X_1: {X_1.shape}')
        print(f'X_2: {X_2.shape}')
        print(f'X_3: {X_3.shape}')

        fig, axs = plt.subplots(3)
        fig.suptitle('NSGT sliced exploration')

        axs[0].plot(X_1)
        axs[0].set_title('original slicq')

        axs[1].plot(X_2)
        axs[1].set_title('max pooled temporally')

        axs[2].plot(X_3)
        axs[2].set_title('reconstructed from maxpool')
        #axs[2].label(f'reconstruction error: {rec_err}')

        [ax.grid() for ax in axs]

        plt.show()

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

        P[name] = torch.abs(src_coef)

        # store the original, not magnitude, in the mix
        model += src_coef

    # now performs separation
    estimates = {}
    for name, source in track.sources.items():
        source_mag = P[name]

        #print('inverting phase')
        Yj = torch.view_as_complex(phasemix_sep(torch.view_as_real(model), source_mag))

        # invert to time domain
        target_estimate = tf.backward(Yj, N)

        # set this as the source estimate
        estimates[name] = target_estimate

    return estimates, time_seq_steps


class TFTransform:
    def __init__(self, fs, fscale="bark", fmin=78.0, fbins=125, fgamma=25.0, sllen=None, device="cuda"):
        self.fbins = fbins
        self.nsgt = None
        self.device = device
        self.fs = fs

        scl = None
        if fscale == 'mel':
            scl = MelScale(fmin, fs/2, fbins)
        elif fscale == 'bark':
            scl = BarkScale(fmin, fs/2, fbins)
        elif fscale == 'cqlog':
            scl = LogScale(fmin, fs/2, fbins)
        elif fscale == 'vqlog':
            scl = VQLogScale(fmin, fs/2, fbins, gamma=fgamma)
        else:
            raise ValueError(f"unsupported scale {fscale}")

        if sllen is None:
            # use slice length required to support desired frequency scale/q factors
            sllen = scl.suggested_sllen(fs)

        self.sllen = sllen
        trlen = sllen//4
        trlen = trlen + -trlen % 2 # make trlen divisible by 2
        self.trlen = trlen

        self.nsgt = NSGT_sliced(scl, sllen, trlen, fs, real=True, matrixform=True, multichannel=True, device=self.device)
        self.name = f'n{fscale}-{fbins}-{fmin:.2f}-{sllen}'

    def forward(self, audio):
        audio = torch.tensor(audio.T, device=self.device)
        audio = torch.unsqueeze(audio, 0)
        C = self.nsgt.forward(audio)
        return C

    def backward(self, X, len_x):
        c = self.nsgt.backward(X, len_x).T
        c = torch.squeeze(c, 0)
        return c

    def printinfo(self):
        print('nsgt params:\n\t{0}\n\t{1} f bins, {2} m bins\n\t{3} total dim'.format(self.name, self.fbins, self.nsgt.ncoefs, self.fbins*self.nsgt.ncoefs))


class TrackEvaluator:
    def __init__(self, tracks, seq_dur_min, seq_dur_max, max_sllen, device="cuda"):
        self.tracks = tracks
        self.seq_dur_min = seq_dur_min
        self.seq_dur_max = seq_dur_max
        self.max_sllen = max_sllen
        self.device = device

    def oracle(self, scale='cqlog', fmin=20.0, bins=12, gamma=25, sllen=None, reps=1, printinfo=False, plot=False):
        bins = int(bins)

        tf = TFTransform(44100, scale, fmin, bins, gamma, sllen=sllen, device=self.device)

        if printinfo:
            tf.printinfo()

        # skip too big transforms
        if tf.sllen > self.max_sllen:
            return (
                float('-inf'),
                tf.sllen
            )

        med_sdrs = []
        mean_coefs = []

        for _ in range(reps):
            # repeat for reps x duration
            for track in self.tracks:
                seq_dur = np.random.uniform(self.seq_dur_min, self.seq_dur_max)
                track.chunk_duration = seq_dur
                track.chunk_start = random.uniform(0, track.duration - seq_dur)

                #print(f'track:\n\t{track.name}\n\t{track.chunk_duration}\n\t{track.chunk_start}')

                N = track.audio.shape[0]
                ests, time_seq_steps = ideal_mixphase(track, tf, plot=plot)

                # maximize score-per-coefficient
                med_sdrs.append(fast_sdr(track, ests, device=self.device)/time_seq_steps)

        # return 1 sdr per source
        return (
            torch.median(torch.cat([torch.unsqueeze(med_sdr, dim=0) for med_sdr in med_sdrs], dim=0)),
            tf.sllen,
        )


def evaluate_single(f, params, seq_reps):
    #print(f'{scale} {bins} {fmin} {fmax} {gamma}')

    curr_score, sllen = f(scale=params['scale'], fmin=params['fmin'], bins=params['bins'], gamma=params['gamma'], sllen=params['sllen'], reps=seq_reps, printinfo=True, plot=True)

    print('total sdr! {0:.2f}'.format(
        curr_score,
    ))


def optimize_many(f, params, n_iter, seq_reps, per_target):
    best_score = float('-inf')
    best_param = None

    fmins = list(np.arange(*params['fmin']))
    gammas = list(np.arange(*params['gamma']))

    #print(f'optimizing target {target_name}')
    for _ in tqdm(range(n_iter)):
        while True: # loop in case we skip for exceeding sllen
            scale = random.choice(params['scales'])
            bins = np.random.randint(*params['bins'])
            fmin = random.choice(fmins)
            gamma = random.choice(gammas)
            
            curr_score, sllen = f(scale=scale, fmin=fmin, bins=bins, gamma=gamma, reps=seq_reps)

            params_tup = (scale, bins, fmin, gamma, sllen)

            if curr_score == float('-inf'):
                # sllen not supported
                print('reroll for sllen...')
                continue

            if curr_score > best_score:
                best_score = curr_score
                best_param = params_tup
                print('good total sdr! {0}, {1}'.format(best_score, best_param))
            break
    print(f'best scores')
    print(f'total: \t{best_score}\t{best_param}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Search NSGT configs for best ideal mask'
    )
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
        default=1000,
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
        '--n-random-tracks',
        type=int,
        default=10,
        help='use N random tracks'
    )
    parser.add_argument(
        '--seq-dur-min',
        type=float,
        default=5.0,
        help='sequence duration per track, min'
    )
    parser.add_argument(
        '--seq-dur-max',
        type=float,
        default=10.0,
        help='sequence duration per track, max'
    )
    parser.add_argument(
        '--seq-reps',
        type=int,
        default=10,
        help='sequence repetitions (adds robustness)'
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
        default=32760,
        help='maximum sllen above which to skip iterations'
    )
    parser.add_argument(
        '--sllen',
        type=int,
        default=8192,
        help='sllen to use'
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

    # initiate musdb
    mus = musdb.DB(subsets='train', split='valid', is_wav=True)

    print(f'using {args.n_random_tracks} random tracks from MUSDB18-HQ train set validation split')
    tracks = random.sample(mus.tracks, args.n_random_tracks)

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

        t = TrackEvaluator(tracks, args.seq_dur_min, args.seq_dur_max, args.max_sllen, device=args.device)
        optimize_many(t.oracle, params, args.n_iter, args.seq_reps, args.per_target)
    else:
        params = {
            'scale': args.fscale,
            'bins': int(args.bins),
            'fmin': float(args.fmins),
            'gamma': float(args.gammas),
            'sllen': int(args.sllen),
        }

        print(f'Parameter to evaluate:\n\t{params}')

        t = TrackEvaluator(tracks, args.seq_dur_min, args.seq_dur_max, args.max_sllen, device=args.device)
        evaluate_single(t.oracle, params, args.seq_reps)
