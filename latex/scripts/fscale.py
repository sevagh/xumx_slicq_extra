import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import librosa
from nsgt.fscale import OctScale, LogScale
import numpy, sys

Fmin = 82.41 # hz of E2
Fmax = 7902.13 # hz of B8

B = int(sys.argv[1]) # user argument for bpo or bins-per-octave
K = int(numpy.ceil(B*numpy.log2(Fmax/Fmin)+1)) # bpo to bins

oct_scale = OctScale(Fmin, Fmax, B) # bins-per-octave param
log_scale = LogScale(Fmin, Fmax, K) # total bins param

print(f'bpo: {B}, bins: {K}, len(oct): {len(oct_scale)}, len(log): {len(log_scale)}')

# call the scale object to get Frequencies and Qs for each bin
oct_fs, oct_Qs = oct_scale()
log_fs, log_Qs = log_scale()

pitches = [librosa.hz_to_note(freq) for freq in oct_fs]
fig, axes = plt.subplots(1)

plt.rcParams.update({'font.size': 14})
axes.plot(oct_fs, 'r--', alpha=0.5, marker='+', markersize=12)
axes.plot(log_fs, 'b:', alpha=0.5, marker='x', markersize=12)
axes.legend([f'oct, {B} bpo', f'log, {K} bins'])
axes.set_title(f'Frequencies for bins-per-octave: {B}, total bins: {K}, {Fmin}-{Fmax} Hz')
axes.set_xlabel('frequency bin')
axes.set_ylabel('frequency (Hz)')

for item in ([axes.title, axes.xaxis.label, axes.yaxis.label] +
             axes.get_xticklabels() + axes.get_yticklabels() + axes.get_legend().get_texts()):
    item.set_fontsize(24)

for i, pitch in enumerate(pitches):
    axes.annotate(pitch, (i, oct_fs[i]))

axes.grid()
axes.xaxis.set_major_locator(MaxNLocator(integer=True))

plt.show()
