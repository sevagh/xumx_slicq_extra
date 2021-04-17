import musdb
import museval
import numpy as np
import functools
import argparse
from scipy.signal import stft, istft


def IBM(track, alpha=1, theta=0.5, eval_dir=None):
    """Ideal Binary Mask:
    processing all channels inpependently with the ideal binary mask.

    the mix is send to some source if the spectrogram of that source over that
    of the mix is greater than theta, when the spectrograms are take as
    magnitude of STFT raised to the power alpha. Typical parameters involve a
    ratio of magnitudes (alpha=1) and a majority vote (theta = 0.5)
    """

    # parameters for STFT
    nfft = 2048

    # small epsilon to avoid dividing by zero
    eps = np.finfo(np.float).eps

    # compute STFT of Mixture
    N = track.audio.shape[0]  # remember number of samples for future use
    X = stft(track.audio.T, nperseg=nfft)[-1]
    (I, F, T) = X.shape

    # perform separtion
    estimates = {}
    accompaniment_source = 0
    for name, source in track.sources.items():

        # compute STFT of target source
        Yj = stft(source.audio.T, nperseg=nfft)[-1]

        # Create Binary Mask
        Mask = np.divide(np.abs(Yj)**alpha, (eps + np.abs(X)**alpha))
        Mask[np.where(Mask >= theta)] = 1
        Mask[np.where(Mask < theta)] = 0

        # multiply mask
        Yj = np.multiply(X, Mask)

        # inverte to time domain and set same length as original mixture
        target_estimate = istft(Yj)[1].T[:N, :]

        # set this as the source estimate
        estimates[name] = target_estimate

        # accumulate to the accompaniment if this is not vocals
        if name != 'vocals':
            accompaniment_source += target_estimate

    # set accompaniment source
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
        description='Evaluate Ideal Binary Mask'
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
        default=1,
        help='exponent for the ratio Mask'
    )

    parser.add_argument(
        '--theta',
        type=float,
        default=0.5,
        help='threshold parameter'
    )

    args = parser.parse_args()

    # default parameters
    alpha = args.alpha
    theta = args.theta

    # initiate musdb
    mus = musdb.DB()

    mus.run(
        functools.partial(
            IBM, alpha=alpha, theta=theta, eval_dir=args.eval_dir
        ),
        estimates_dir=args.audio_dir,
        subsets='test',
        parallel=True,
        cpus=2
    )
