from nsgt import NSGT, NSGT_sliced
from nsgt.fscale import LogScale
import numpy

Fmin = 82.41 # hz of E2
Fmax = 7902.13 # hz of B8

K = 48    # high total bins
fs = 44100 # musdb18-hq's cd sample rate

log_scale = LogScale(Fmin, Fmax, K)

# test audio waveform - 2s of random samples in [-1.0, 1.0]
test_signal = numpy.random.uniform(-1,1,fs*2)

# length of total signal
Ls = len(test_signal)

# nsgt takes the whole input signal length
nsgt = NSGT(log_scale, fs, Ls)

# sliced nsgt with inappropriate sllen, trlen = 2048, 512
slicqt_warning = NSGT_sliced(log_scale, 2048, 512, fs)

# finally, get the suggested sllen and trlen for a sample rate
sllen, trlen = log_scale.suggested_sllen_trlen(fs)
print(f'suggested sllen, trlen: {sllen} {trlen}')

slicqt_no_warning = NSGT_sliced(log_scale, sllen, trlen, fs)
