import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import librosa
import numpy, sys

Fmin = 20    # lower limit of human hearing
Fmax = 22050 # nyquist of musdb18hq's 44100/cd sampling rate

K = int(sys.argv[1]) # user argument for total bins

nfft=512

mels = librosa.filters.mel(sr=44100, n_mels=K, n_fft=nfft, fmin=Fmin, fmax=Fmax)
mels /= numpy.max(mels, axis=-1)[:, None]

barks = librosa.filters.bark(sr=44100, n_barks=K, n_fft=nfft, fmin=Fmin, fmax=Fmax)
barks /= numpy.max(barks, axis=-1)[:, None]

fig, axes = plt.subplots(2, 1)

plt.rcParams.update({'font.size': 14})

#axes[0].set_yscale('log')

axes[0].plot(mels.T)
axes[0].set_title(f'{int(nfft/2+1)} frequencies distributed into {K} mel bands, {Fmin}-{Fmax} Hz')
axes[0].set_xlabel('frequency bin')
axes[0].set_ylabel('normalized weight')

for item in ([axes[0].title, axes[0].xaxis.label, axes[0].yaxis.label] +
             axes[0].get_xticklabels() + axes[0].get_yticklabels()):
    item.set_fontsize(24)

axes[0].grid()
axes[0].xaxis.set_major_locator(MaxNLocator(integer=True))

axes[1].plot(barks.T)
axes[1].set_title(f'{int(nfft/2+1)} frequencies distributed into {K} Bark bands, {Fmin}-{Fmax} Hz')
axes[1].set_xlabel('frequency bin')
axes[1].set_ylabel('normalized weight')

for item in ([axes[1].title, axes[1].xaxis.label, axes[1].yaxis.label] +
             axes[1].get_xticklabels() + axes[1].get_yticklabels()):
    item.set_fontsize(24)

axes[1].grid()
axes[1].xaxis.set_major_locator(MaxNLocator(integer=True))

#fig.tight_layout()
plt.subplots_adjust(hspace=0.35)
plt.show()
