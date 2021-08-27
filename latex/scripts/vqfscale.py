import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import librosa
from nsgt.fscale import OctScale, LogScale, VQLogScale
import numpy, sys

Fmin = 82.41 # hz of E2
Fmax = 7902.13

K = int(sys.argv[1]) # user argument for total bins

log_scale = LogScale(Fmin, Fmax, K) # total bins param

gammas = [3, 6.6, 10, 30] # hz
vql_scales = [VQLogScale(Fmin, Fmax, K, gamma=gamma) for gamma in gammas] # total bins param

# call the scale object to get Frequencies and Qs for each bin
log_fs, log_Qs = log_scale()

pitches = [librosa.hz_to_note(freq) for freq in log_fs]
fig, axes = plt.subplots(2, 1)

plt.rcParams.update({'font.size': 14})
axes[0].plot(log_fs, 'r--', alpha=0.5, marker='+', markersize=12)

axes[0].set_yscale('log')

for i, gamma in enumerate(gammas):
    if i == 0:
        linestr = 'm-'
    elif i == 1:
        linestr = 'c-.'
    elif i == 2:
        linestr = 'b:'
    elif i == 3:
        linestr = 'g'

    gamma_fs, gamma_qs = vql_scales[i]()
    print(gamma_fs)
    axes[0].plot(gamma_fs, linestr, alpha=0.5, marker='x', markersize=12)

axes[0].legend([f'cq/log, gamma=0'] + [f'vq, gamma={gamma}' for gamma in gammas])
axes[0].set_title(f'Frequencies for total bins: {K}, {Fmin}-{Fmax} Hz')
axes[0].set_xlabel('frequency bin')
axes[0].set_ylabel('frequency (Hz)')

for item in ([axes[0].title, axes[0].xaxis.label, axes[0].yaxis.label] +
             axes[0].get_xticklabels() + axes[0].get_yticklabels() + axes[0].get_legend().get_texts()):
    item.set_fontsize(24)

for i, pitch in enumerate(pitches):
    axes[0].annotate(pitch, (i, numpy.log2(log_fs[i])))

axes[0].grid()
axes[0].xaxis.set_major_locator(MaxNLocator(integer=True))

axes[1].plot(log_Qs, 'r--', alpha=0.5, marker='+', markersize=12)

for i, gamma in enumerate(gammas):
    if i == 1:
        linestr = 'm-'
    elif i == 1:
        linestr = 'c-.'
    elif i == 2:
        linestr = 'b:'
    elif i == 3:
        linestr = 'g'

    gamma_fs, gamma_Qs = vql_scales[i]()
    axes[1].plot(gamma_Qs, linestr, alpha=0.5, marker='x', markersize=12)

axes[1].legend([f'cq/log, gamma=0'] + [f'vq, gamma={gamma}' for gamma in gammas])
axes[1].set_title(f'Q-factors for total bins: {K}, {Fmin}-{Fmax} Hz')
axes[1].set_xlabel('frequency bin')
axes[1].set_ylabel('Q-factor (ratio)')

for item in ([axes[1].title, axes[1].xaxis.label, axes[1].yaxis.label] +
             axes[1].get_xticklabels() + axes[1].get_yticklabels() + axes[1].get_legend().get_texts()):
    item.set_fontsize(24)

axes[1].grid()
axes[1].xaxis.set_major_locator(MaxNLocator(integer=True))

#fig.tight_layout()
plt.subplots_adjust(hspace=0.35)
plt.show()
