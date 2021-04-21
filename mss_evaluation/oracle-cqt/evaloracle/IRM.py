import sys
import os
import musdb
import gc
import itertools
import museval
import numpy as np
import functools
import argparse
from scipy.signal import stft, istft

# use CQT based on nonstationary gabor transform
from nsgt import CQ_NSGT


def stereo_nsgt(audio, nsgt):
    print(audio.shape)
    X_chan1 = np.asarray(nsgt.forward(audio[:, 0]))
    X_chan2 = np.asarray(nsgt.forward(audio[:, 1]))
    X = np.empty((2, X_chan1.shape[0], X_chan1.shape[1]), dtype=np.complex)
    X[0, :, :] = X_chan1
    X[1, :, :] = X_chan2
    print(X.shape)
    return X


def stereo_insgt(C, nsgt):
    print(C.shape)
    C_chan1 = C[0, :, :]
    C_chan2 = C[1, :, :]
    inv1 = nsgt.backward(C_chan1)
    inv2 = nsgt.backward(C_chan2)
    ret_audio = np.empty((2, inv1.shape[0]), dtype=np.float32)
    ret_audio[0, :] = inv1
    ret_audio[1, :] = inv2
    print(ret_audio.shape)
    return ret_audio.T


class TFTransform:
    def __init__(self, ntrack, fs, nfft=2048, use_cqt=False, fmin=27.5, cqt_bins=96):
        if use_cqt:
            print(f'using CQT with params: {fmin=}, {cqt_bins=}')
        else:
            print(f'using STFT with nfft: {nfft=}')

        self.nsgt = None
        self.nfft = nfft
        self.N = ntrack
        self.cq_nsgt = None
        if use_cqt:
            # use nyquist as maximum frequency always
            # nsgt has a multichannel=True param which blows memory up. prefer to do it myself
            self.cq_nsgt = CQ_NSGT(fmin, fs/2, cqt_bins, fs, self.N, matrixform=True, multithreading=True)

    def forward(self, audio):
        if not self.cq_nsgt:
            print('forward stft')
            return stft(audio.T, nperseg=self.nfft)[-1]
        else:
            print('forward cq-nsgt')
            return stereo_nsgt(audio, self.cq_nsgt)

    def backward(self, X):
        if not self.cq_nsgt:
            print('backward stft')
            return istft(X)[1].T[:self.N, :]
        else:
            print('backward cq-nsgt')
            return stereo_insgt(X, self.cq_nsgt)


def ideal_mask(track, alpha=2, binary_mask=False, theta=0.5, eval_dir=None, stft_nperseg=2048, use_cqt=False, fmin=27.5, cqt_bins=96):
    """
    if theta=None:
        Ideal Ratio Mask:
        processing all channels inpependently with the ideal ratio mask.
        this is the ratio of spectrograms, where alpha is the exponent to take for
        spectrograms. usual values are 1 (magnitude) and 2 (power)

    if theta=float:
        Ideal Binary Mask:
        processing all channels inpependently with the ideal binary mask.

        the mix is send to some source if the spectrogram of that source over that
        of the mix is greater than theta, when the spectrograms are take as
        magnitude of STFT raised to the power alpha. Typical parameters involve a
        ratio of magnitudes (alpha=1) and a majority vote (theta = 0.5)
    """

    # small epsilon to avoid dividing by zero
    eps = np.finfo(np.float).eps

    # compute STFT of Mixture
    N = track.audio.shape[0]  # remember number of samples for future use

    tf = TFTransform(N, track.rate, stft_nperseg, use_cqt, fmin, cqt_bins)

    X = tf.forward(track.audio)
    (I, F, T) = X.shape

    # soft mask stuff
    if not binary_mask:
        # Compute sources spectrograms
        P = {}
        # compute model as the sum of spectrograms
        model = eps

        for name, source in track.sources.items():
            # compute spectrogram of target source:
            # magnitude of STFT to the power alpha
            P[name] = np.abs(tf.forward(source.audio))**alpha
            model += P[name]

    # now performs separation
    estimates = {}
    accompaniment_source = 0
    for name, source in track.sources.items():
        if binary_mask:
            print('binary/hard mask')
            # compute STFT of target source
            Yj = tf.forward(source.audio)

            # Create Binary Mask
            Mask = np.divide(np.abs(Yj)**alpha, (eps + np.abs(X)**alpha))
            Mask[np.where(Mask >= theta)] = 1
            Mask[np.where(Mask < theta)] = 0
        else:
            print('ratio/soft mask')
            # compute soft mask as the ratio between source spectrogram and total
            Mask = np.divide(np.abs(P[name]), model)

        # multiply the mix by the mask
        Yj = np.multiply(X, Mask)

        # invert to time domain
        target_estimate = tf.backward(Yj)

        # set this as the source estimate
        estimates[name] = target_estimate

        # accumulate to the accompaniment if this is not vocals
        if name != 'vocals':
            accompaniment_source += target_estimate

    estimates['accompaniment'] = accompaniment_source

    if eval_dir is not None:
        museval.eval_mus_track(
            track,
            estimates,
            output_dir=eval_dir,
        )

    return estimates


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Evaluate Ideal Ratio Mask'
    )
    parser.add_argument(
        '--audio_dir',
        nargs='?',
        help='Folder where audio results are saved',
        default=None,
    )

    parser.add_argument(
        '--eval_dir',
        nargs='?',
        help='Folder where evaluation results are saved'
    )

    parser.add_argument(
        '--alpha',
        type=int,
        default=2,
        help='exponent for the ratio Mask'
    )
    parser.add_argument(
        '--use-cqt',
        action='store_true',
        help='use NSGT-CQT instead of STFT'
    )
    parser.add_argument(
        '--binary-mask',
        action='store_true',
        help='use binary mask instead of soft'
    )
    parser.add_argument(
        '--binary-theta',
        type=float,
        default=0.5,
        help='theta/separation factor for binary mask',
    )
    parser.add_argument(
        '--stft-nperseg',
        default=2048,
        type=int,
        help='stft nperseg'
    )
    parser.add_argument(
        '--cqt-fmin',
        default=27.5,
        type=float,
        help='NSGT-CQT minimum frequency'
    )
    parser.add_argument(
        '--cqt-bins',
        default=96,
        type=int,
        help='NSGT-CQT total bins'
    )

    args = parser.parse_args()

    max_tracks = int(os.getenv('MUSDB_MAX_TRACKS', sys.maxsize))

    # initiate musdb
    mus = musdb.DB(subsets='test', is_wav=True)

    for track in mus.tracks[:max_tracks]:
        est = ideal_mask(
            track,
            args.alpha,
            args.binary_mask,
            args.binary_theta,
            args.eval_dir,
            args.stft_nperseg,
            args.use_cqt,
            args.cqt_fmin,
            args.cqt_bins,
        )
        gc.collect()
        if args.audio_dir:
            mus.save_estimates(est, track, args.audio_dir)
