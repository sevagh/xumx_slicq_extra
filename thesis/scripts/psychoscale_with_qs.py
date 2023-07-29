import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import librosa
from nsgt.fscale import MelScale, BarkScale
import numpy, sys

Fmin = 82.41 # hz of E2
Fmax = 7902.13 # hz of B8

K = int(sys.argv[1]) # total bins argument

mel_scale = MelScale(Fmin, Fmax, K)
bark_scale = BarkScale(Fmin, Fmax, K)

# call the scale object to get Frequencies and Qs for each bin
mel_fs, mel_Qs = mel_scale()
bark_fs, bark_Qs = bark_scale()

#mel_pitches = [librosa.hz_to_note(freq) for freq in mel_fs]
#bark_pitches = [librosa.hz_to_note(freq) for freq in bark_fs]
fig, axes = plt.subplots(2, 1)

plt.rcParams.update({'font.size': 14})

axes[0].plot(mel_fs, 'r--', alpha=0.5, marker='+', markersize=12)
axes[0].plot(bark_fs, 'b:', alpha=0.5, marker='x', markersize=12)
axes[0].legend(['mel', 'bark'])
axes[0].set_title(f'Frequencies for total bins: {K}, {Fmin}-{Fmax} Hz')
axes[0].set_xlabel('frequency bin')
axes[0].set_ylabel('frequency (Hz)')

for item in ([axes[0].title, axes[0].xaxis.label, axes[0].yaxis.label] +
             axes[0].get_xticklabels() + axes[0].get_yticklabels() + axes[0].get_legend().get_texts()):
    item.set_fontsize(24)

#for i, pitch in enumerate(mel_pitches):
#    axes[0].annotate(pitch, (i, mel_fs[i]), xytext=(i, mel_fs[i]-750.0))
#
#for i, pitch in enumerate(bark_pitches):
#    axes[0].annotate(pitch, (i, bark_fs[i]), xytext=(i, bark_fs[i]+750.0))

axes[0].grid()
axes[0].xaxis.set_major_locator(MaxNLocator(integer=True))

axes[1].plot(mel_Qs, 'r--', alpha=0.5, marker='+', markersize=12)
axes[1].plot(bark_Qs, 'b:', alpha=0.5, marker='x', markersize=12)
axes[1].legend(['mel', 'bark'])
axes[1].set_title(f'Q-factors for total bins: {K}, {Fmin}-{Fmax} Hz')
axes[1].set_xlabel('frequency bin')
axes[1].set_ylabel('Q-factor (ratio)')

for item in ([axes[1].title, axes[1].xaxis.label, axes[1].yaxis.label] +
             axes[1].get_xticklabels() + axes[1].get_yticklabels() + axes[1].get_legend().get_texts()):
    item.set_fontsize(24)

axes[1].grid()
axes[1].xaxis.set_major_locator(MaxNLocator(integer=True))

plt.subplots_adjust(hspace=0.35)
plt.show()
