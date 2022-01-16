import os
from warnings import warn
import torch
import numpy
from slicqt.plot import spectrogram
from slicqt.audio import SndReader
from slicqt.torch_utils import load_deoverlapnet
import auraloss

from slicqt import torch_transforms as transforms

from argparse import ArgumentParser
import matplotlib.pyplot as plt


if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument("input", type=str, help="Input file")
    parser.add_argument("--deoverlapnet-path", type=str, default="./pretrained-model")
    parser.add_argument("--output-png", type=str, help="output png path", default=None)
    parser.add_argument("--sr", type=int, default=44100, help="Sample rate used for the NSGT (default=%(default)s)")
    parser.add_argument("--cmap", type=str, default='hot', help="spectrogram color map")
    parser.add_argument("--fontsize", type=int, default=14, help="Plot font size, default=%(default)s)")
    parser.add_argument("--plot", action="store_true", help="Plot spectrogram")

    args = parser.parse_args()
    if not os.path.exists(args.input):
        parser.error("Input file '%s' not found"%args.input)

    print('loading deoverlapnet...')
    deoverlapnet, slicqt, islicqt = load_deoverlapnet(args.deoverlapnet_path)

    fs = args.sr

    # Read audio data
    sf = SndReader(args.input, sr=fs, chns=2)
    signal = sf()

    signal = [torch.tensor(sig) for sig in signal]
    signal = torch.cat(signal, dim=-1)

    # add a batch
    signal = torch.unsqueeze(signal, dim=0)

    # duration of signal in s
    dur = sf.frames/float(fs)

    C = slicqt(signal)

    Cmag, C_phase = transforms.complex_2_magphase(C)
    Cmag_ola = slicqt.overlap_add(Cmag)
    nb_slices = Cmag[0].shape[-2]

    ragged_ola_shapes = [Cmag_ola_.shape for Cmag_ola_ in Cmag_ola]

    Cmag_hat = deoverlapnet(Cmag_ola, nb_slices, ragged_ola_shapes)
    C_hat = transforms.magphase_2_complex(Cmag_hat, C_phase)

    signal_recon = islicqt(C_hat, signal.shape[-1])

    print('Error measures:')

    print(f'\tSNR: {auraloss.time.SNRLoss()(signal_recon, signal)}')
    print(f'\tMSE (magnitude sliCQT): {sum([torch.sqrt(torch.mean((Cmag_hat[i] - Cmag[i])**2)) for i in range(len(Cmag))])/len(Cmag)}')
    print(f'\tMSE (time-domain waveform): {torch.sqrt(torch.mean((signal_recon - signal)**2))}')
    print(f'\tSI-SDR: {auraloss.time.SISDRLoss()(signal_recon, signal)}')
    print(f'\tSD-SDR: {auraloss.time.SDSDRLoss()(signal_recon, signal)}')

    if args.plot:
        freqs = slicqt.nsgt.freqs
        if slicqt.nsgt.fmin > 0.0:
            freqs = numpy.r_[[0.], freqs]

        spectrogram(
            Cmag_interp_ola.detach(),
            fs,
            slicqt.nsgt.M,
            slicqt.nsgt.nsgt.coef_factor,
            freqs,
            signal.shape[-1],
            fontsize=args.fontsize,
            cmap=args.cmap,
            slicq_name=args.slicq_config,
            output_file=args.output_png
        )
