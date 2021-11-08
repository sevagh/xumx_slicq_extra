from nsgt import NSGT, NSGT_sliced
from nsgt.fscale import LogScale
import numpy

Fmin = 82.41 # hz of E2
Fmax = 7902.13 # hz of B8

K = 48    # high total bins
fs = 44100 # musdb18-hq's cd sample rate

log_scale = LogScale(Fmin, Fmax, K)
test_signal = numpy.random.uniform(-1,1,fs*2) # 2s of random audio

Ls = len(test_signal) # length of total signal

sllen, trlen = log_scale.suggested_sllen_trlen(fs) # get best suggested sllen/trlen
print(f'suggested sllen, trlen: {sllen} {trlen}')

slicqt = NSGT_sliced(log_scale, sllen, trlen, fs) # slicqt with good sllen/trlen
