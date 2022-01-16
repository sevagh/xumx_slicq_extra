import os
from warnings import warn
import torch
import numpy
from slicqt.plot import spectrogram
from slicqt.torch_transforms import SliCQTBase, make_filterbanks, complex_2_magphase
from slicqt.audio import SndReader
from slicqt.fscale import Pow2Scale, LogScale, LinScale, MelScale, OctScale, VQLogScale, BarkScale

from argparse import ArgumentParser
import matplotlib.pyplot as plt


if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument("input", type=str, help="Input file")
    parser.add_argument("--output", type=str, help="output png path", default=None)
    parser.add_argument("--sr", type=int, default=44100, help="Sample rate used for the NSGT (default=%(default)s)")
    parser.add_argument("--fmin", type=float, default=50, help="Minimum frequency in Hz (default=%(default)s)")
    parser.add_argument("--fmax", type=float, default=22050, help="Maximum frequency in Hz (default=%(default)s)")
    parser.add_argument("--gamma", type=float, default=15, help="variable-q frequency offset per band")
    parser.add_argument("--cmap", type=str, default='hot', help="spectrogram color map")
    parser.add_argument("--scale", choices=('oct','cqlog','mel','bark','Bark','vqlog','pow2'), default='cqlog', help="Frequency scale")
    parser.add_argument("--bins", type=int, default=50, help="Number of frequency bins (total or per octave, default=%(default)s)")
    parser.add_argument("--fontsize", type=int, default=14, help="Plot font size, default=%(default)s)")
    parser.add_argument("--plot", action='store_true', help="Plot transform (needs installed matplotlib package)")

    args = parser.parse_args()
    if not os.path.exists(args.input):
        parser.error("Input file '%s' not found"%args.input)

    fs = args.sr

    # Read audio data
    sf = SndReader(args.input, sr=fs, chns=2)
    signal = sf()

    signal = [torch.tensor(sig) for sig in signal]
    signal = torch.cat(signal, dim=-1)

    # add a batch of 1 for samples
    signal = torch.unsqueeze(signal, dim=0)

    # duration of signal in s
    dur = sf.frames/float(fs)

    slicq_base = SliCQTBase(
        args.scale,
        args.bins,
        args.fmin,
        fmax=args.fmax,
        fs=args.sr,
        gamma=args.gamma,
        device="cpu",
    )

    slicq, islicq = make_filterbanks(slicq_base)

    freqs, qs = slicq_base.scl()

    C = slicq.forward(signal)
    Cmag, Cphase = complex_2_magphase(C)

    Cmag_interp = slicq.interpolate(Cmag)
    Cmag_interp_ola = slicq.overlap_add(Cmag_interp)

    print(f'type Cmag: {type(Cmag)}, {Cmag[0].shape}')
    print(f'type Cphase: {type(Cphase)}, {Cphase[0].shape}')
    print(f'type Cmag_interp: {type(Cmag_interp)}, {Cmag_interp.shape}')
    print(f'type Cmag_interp_ola: {type(Cmag_interp_ola)}, {Cmag_interp_ola.shape}')

    if args.fmin > 0.0:
        freqs = numpy.r_[[0.], freqs]

    if args.plot:
        slicq_params = '{0} scale, {1} bins, {2:.1f}-{3:.1f} Hz'.format(args.scale, args.bins, args.fmin, args.fmax)

        spectrogram(
            Cmag_interp_ola,
            fs,
            slicq.nsgt.M,
            slicq.nsgt.nsgt.coef_factor,
            freqs,
            signal.shape[-1],
            fontsize=args.fontsize,
            cmap=args.cmap,
            slicq_name=slicq_params,
            output_file=args.output
        )
