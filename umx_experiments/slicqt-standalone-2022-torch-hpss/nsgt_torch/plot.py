import matplotlib.pyplot as plt
import torch
import numpy


def spectrogram(c, fs, chop, coef_factor, transform_name, freqs, frames, sliced=True, fontsize=14, cmap='inferno', slicq_name='', output_file=None, flattened=False):
    if type(c) != torch.Tensor:
        raise ValueError('only use this with interpolated_matrixform')

    mls = 20.*torch.log10(c) # dB

    if sliced:
        mls = mls[:, :, :, int(chop/2):]
        mls = mls[:, :, :, :-int(chop/2)]

    plt.rcParams.update({'font.size': fontsize})
    fig, axs = plt.subplots(1)

    print(f"Plotting t*f space")

    # remove batch
    mls = torch.squeeze(mls, dim=0)
    # mix down multichannel
    mls = torch.mean(mls, dim=0)

    mls = mls.T

    fs_coef = fs*coef_factor # frame rate of coefficients

    ncoefs = int(coef_factor*frames)
    if flattened:
        ncoefs *= 2

    mls = mls[:ncoefs, :]

    mls_dur = len(mls)/fs_coef # final duration of MLS

    nb_bins = len(freqs)

    mls_max = torch.quantile(mls, 0.999)

    dur_for_plot = mls_dur
    if flattened:
        # doctor the duration to accomodate the lack of overlap-adding the slices
        dur_for_plot /= 2

    try:
        im = axs.pcolormesh(numpy.linspace(0.0, dur_for_plot, num=ncoefs), freqs/1000., mls.T, vmin=mls_max-120., vmax=mls_max, cmap=cmap)
    except TypeError:
        freqs = numpy.r_[[0.], freqs]
        im = axs.pcolormesh(numpy.linspace(0.0, dur_for_plot, num=ncoefs), freqs/1000., mls.T, vmin=mls_max-120., vmax=mls_max, cmap=cmap)

    title = f'Magnitude {transform_name}'

    if slicq_name != '':
        title += f', {slicq_name}'

    axs.set_title(title)

    axs.set_xlabel('Time (s)')
    axs.set_ylabel('Frequency (kHz)')
    axs.yaxis.get_major_locator().set_params(integer=True)

    fig.colorbar(im, ax=axs, shrink=1.0, pad=0.006, label='dB')

    plt.subplots_adjust(wspace=0.001,hspace=0.001)

    if output_file is not None:
        DPI = fig.get_dpi()
        fig.set_size_inches(2560.0/float(DPI),1440.0/float(DPI))
        fig.savefig(output_file, dpi=DPI, bbox_inches='tight')
    else:
        plt.show()
