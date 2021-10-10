import random
import numpy as np
import sys
from essentia import lin2db
import matplotlib.pyplot as plt
from sliceq22.overlap import nsgcq_overlap_add


def SNR(r, t, skip=8192):
    """
    r    : reference
    t    : test
    skip : number of samples to skip from the SNR computation
    """
    difference = ((r[skip: -skip] - t[skip: -skip]) ** 2).sum()
    return lin2db((r[skip: -skip] ** 2).sum() / difference)


def extract_segment_randomly(x_full, fs, seq_len):
    dur = float(len(x_full))/float(fs)
    if seq_len > dur:
        return x_full

    seq_len_samples = seq_len*fs
    start = int(np.floor(np.random.uniform(low=0, high=dur-seq_len)))
    end = int(start+seq_len_samples)
    x_seg = x_full[start:end]
    return x_seg


def plot_slicq(x, y_gt, y_pred):
    x = np.squeeze(np.squeeze(x, axis=0), axis=-1)
    y_gt = nsgcq_overlap_add(np.squeeze(np.squeeze(y_gt, axis=0), axis=-1))
    y_pred = nsgcq_overlap_add(np.squeeze(np.squeeze(y_pred, axis=0), axis=-1))

    print(f'x: {x.shape}, y_gt: {y_gt.shape}, y_pred: {y_pred.shape}')

    fig, axs = plt.subplots(3)

    axs[0].matshow(np.log10(np.abs(x)), origin='lower', aspect='auto')
    axs[0].set_title('Magnitude sliCQT (dB), training input')

    axs[1].matshow(np.log10(np.abs(y_gt)), origin='lower', aspect='auto')
    axs[1].set_title('Magnitude sliCQT (dB), ground truth output')

    axs[2].matshow(np.log10(np.abs(y_pred)), origin='lower', aspect='auto')
    axs[2].set_title('Magnitude sliCQT (dB), predicted output')

    plt.tight_layout()
    plt.show()
