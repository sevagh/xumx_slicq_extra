import os
from warnings import warn
import torch
import numpy
from xumx_slicq_22.nsgt.plot import spectrogram
from xumx_slicq_22.nsgt.audio import SndReader
from xumx_slicq_22.utils import load_deoverlapnet
import auraloss

from xumx_slicq_22 import transforms

from argparse import ArgumentParser
import matplotlib.pyplot as plt


parser = ArgumentParser()

parser.add_argument("input", type=str, help="Input file")
parser.add_argument("--output-png", type=str, help="output png path", default=None)
parser.add_argument("--sr", type=int, default=44100, help="Sample rate used for the NSGT (default=%(default)s)")
parser.add_argument("--cmap", type=str, default='hot', help="spectrogram color map")
parser.add_argument("--fontsize", type=int, default=14, help="Plot font size, default=%(default)s)")
parser.add_argument("--slicq-config", choices=('harmonic','percussive'), default='harmonic', help="which of the two slicqs to use")
parser.add_argument("--plot", action="store_true", help="Plot spectrogram")

args = parser.parse_args()
if not os.path.exists(args.input):
    parser.error("Input file '%s' not found"%args.input)

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

nsgt = None
if args.slicq_config == 'harmonic':
    nsgt = transforms.NSGTBase(
        scale='bark',
        fmin=29.8,
        fbins=287,
        fs=fs,
        device="cpu"
    )
elif args.slicq_config == 'percussive':
    nsgt = transforms.NSGTBase(
        scale='bark',
        fmin=33.4,
        fbins=265,
        fs=fs,
        device="cpu"
    )

deoverlapnet = load_deoverlapnet()

slicq, islicq = transforms.make_filterbanks_slicqt(nsgt)

c = slicq(signal)
cnorm = transforms.ComplexNormSliCQT()

c_mag = cnorm(c)
prev_shapes = [c_mag_.shape for c_mag_ in c_mag]
c_interp = slicq.interpolate(c_mag)

max_t_bins = c_interp.shape[-1]
nb_slices = c_interp.shape[-2]

c_interp_ola = slicq.overlap_add(c_interp)

c_mag_pred = None
if args.slicq_config == 'harmonic':
    c_mag_pred, _ = deoverlapnet(harmonic_inputs=(c_interp_ola, nb_slices, prev_shapes))
elif args.slicq_config == 'percussive':
    _, c_mag_pred = deoverlapnet(percussive_inputs=(c_interp_ola, nb_slices, prev_shapes))

c_pred = transforms.phasemix_sep(c, c_mag_pred)

signal_recon = islicq(c_pred, signal.shape[-1])

print('Error measures:')

print(f'\tSNR: {auraloss.time.SNRLoss()(signal_recon, signal)}')
print(f'\tMSE (magnitude sliCQT): {sum([torch.sqrt(torch.mean((c_mag_pred[i] - c_mag[i])**2)) for i in range(len(c_mag))])/len(c_mag)}')
print(f'\tMSE (time-domain waveform): {torch.sqrt(torch.mean((signal_recon - signal)**2))}')
print(f'\tSI-SDR: {auraloss.time.SISDRLoss()(signal_recon, signal)}')
print(f'\tSD-SDR: {auraloss.time.SDSDRLoss()(signal_recon, signal)}')

if args.plot:
    freqs = slicq.nsgt.freqs
    if slicq.nsgt.fmin > 0.0:
        freqs = numpy.r_[[0.], freqs]

    spectrogram(
        c_interp_ola.detach(),
        max_t_bins,
        fs,
        slicq.nsgt.nsgt.coef_factor,
        'sliCQT',
        freqs,
        signal.shape[-1],
        fontsize=args.fontsize,
        cmap=args.cmap,
        slicq_name=args.slicq_config,
        output_file=args.output_png
    )
