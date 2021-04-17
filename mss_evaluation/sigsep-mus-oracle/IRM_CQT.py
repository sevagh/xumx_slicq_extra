import sys
import musdb
import museval
import numpy as np
import functools
import argparse

# use CQT based on nonstationary gabor transform
from nsgt import NSGT, OctScale


def stereo_nsgt(audio, nsgt):
    X_chan1 = np.asarray(nsgt.forward(audio[:, 0]))
    X_chan2 = np.asarray(nsgt.forward(audio[:, 1]))
    X = np.empty((2, X_chan1.shape[0], X_chan1.shape[1]), dtype=np.complex)
    X[0, :, :] = X_chan1
    X[1, :, :] = X_chan2
    return X

def stereo_insgt(C, nsgt):
    C_chan1 = C[0, :, :]
    C_chan2 = C[1, :, :]
    inv1 = nsgt.backward(C_chan1)
    inv2 = nsgt.backward(C_chan2)
    ret_audio = np.empty((2, inv1.shape[0]), dtype=np.float32)
    ret_audio[0, :] = inv1
    ret_audio[1, :] = inv2
    return ret_audio.T


def IRM_CQT(track, alpha=2, eval_dir=None):
    """Ideal Ratio Mask:
    processing all channels inpependently with the ideal ratio mask.
    this is the ratio of spectrograms, where alpha is the exponent to take for
    spectrograms. usual values are 1 (magnitude) and 2 (power)"""

    # CQT parameters
    bins_per_octave = 96
    scl = OctScale(20, 22050, 96)

    # small epsilon to avoid dividing by zero
    eps = np.finfo(np.float).eps

    # compute STFT of Mixture
    N = track.audio.shape[0]  # remember number of samples for future use

    nsgt = NSGT(scl, track.rate, N, real=True, matrixform=True)

    X = stereo_nsgt(track.audio, nsgt)
    (I, F, T) = X.shape

    # Compute sources spectrograms
    P = {}
    # compute model as the sum of spectrograms
    model = eps

    for name, source in track.sources.items():
        # compute spectrogram of target source:
        # magnitude of STFT to the power alpha
        P[name] = np.abs(stereo_nsgt(source.audio, nsgt))**alpha
        model += P[name]

    # now performs separation
    estimates = {}
    accompaniment_source = 0
    for name, source in track.sources.items():
        # compute soft mask as the ratio between source spectrogram and total
        Mask = np.divide(P[name], model)

        # multiply the mix by the mask
        Yj = np.multiply(X, Mask)

        # invert to time domain
        target_estimate = stereo_insgt(Yj, nsgt)

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
        help='Folder where audio results are saved'
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
        '--max-tracks',
        type=int,
        default=sys.maxsize,
        help='maximum tracks'
    )

    args = parser.parse_args()

    alpha = args.alpha

    # initiate musdb
    mus = musdb.DB(subsets='test', is_wav=True)

    for track in mus.tracks[:args.max_tracks]:
        estimates = IRM_CQT(track, alpha=alpha, eval_dir=args.eval_dir)
