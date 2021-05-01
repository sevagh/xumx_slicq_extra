'''
adapted from https://github.com/sigsep/sigsep-mus-2018-analysis/blob/master/sisec-2018-paper-figures/boxplot.py
'''

import pandas
import seaborn
import argparse
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib
import sys
import gc
import itertools

controls = ['ibm1', 'ibm2', 'irm1', 'irm2'] 


def save_boxplot(pandas_in, pdf_out, single=False):
    metrics = ['metrics.SDR', 'metrics.SIR', 'metrics.SAR', 'metrics.ISR']
    targets = ['vocals', 'accompaniment', 'drums', 'bass', 'other']

    df = pandas.read_pickle(pandas_in)
    df['control'] = df.method.isin(controls)

    # aggregate methods by mean using median by track
    df = df.groupby(
        ['method', 'track', 'target', 'metric']
    ).median().reset_index()

    # Get sorting keys (sorted by median of SDR:vocals)
    df_sort_by = df[
        (df.metric == "metrics.SDR") &
        (df.target == "vocals")
    ]

    methods_by_sdr = df_sort_by.score.groupby(
        df_sort_by.method
    ).median().sort_values().index.tolist()

    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    seaborn.set()
    seaborn.set_context("paper")

    params = {
        'backend': 'ps',
        'axes.labelsize': 18,
        'font.size': 15,
        'legend.fontsize': 16,
        'xtick.labelsize': 13,
        'ytick.labelsize': 15,
        'text.usetex': True,
        'font.family': 'serif',
        'font.serif': 'ptmrr8re',
    }

    seaborn.set_style("darkgrid", {
        'pgf.texsystem': 'xelatex',  # pdflatex, xelatex, lualatex
        "axes.facecolor": "0.925",
        'text.usetex': True,
        'font.family': 'serif',
        'axes.labelsize': 14,
        'font.size': 14,
        'legend.fontsize': 15,
        'xtick.labelsize': 15,
        'ytick.labelsize': 17,
        'font.serif': [],
    })
    plt.rcParams.update(params)

    if single:
        with PdfPages(pdf_out) as pdf:
            for (target, metric) in itertools.product(targets, metrics):
                g = seaborn.boxplot(
                    x="score",
                    y="method",
                    hue="control",
                    data=df.loc[(df['target'] == target) & (df['metric'] == metric)],
                    orient='h',
                    order=methods_by_sdr[::-1],
                    hue_order=[True, False],
                    showfliers=False,
                    notch=True,
                )
                g.legend_.remove()
                g.figure.suptitle(f'{target} - {metric}')
                g.figure.set_size_inches(8.5,11)
                g.figure.tight_layout()
                g.figure.savefig(
                    pdf,
                    format='pdf',
                    bbox_inches='tight',
                    dpi=300,
                )
                del g
                gc.collect()
                plt.clf()
    else:
        g = seaborn.FacetGrid(
            df,
            row="target",
            col="metric",
            row_order=targets,
            col_order=metrics,
            size=6,
            sharex=False,
            aspect=0.7
        )
        g = (g.map(
            seaborn.boxplot,
            "score",
            "method",
            orient='h',
            order=methods_by_sdr[::-1],
            hue_order=[True, False],
            showfliers=False,
            notch=True
        ))
        plt.setp(g.fig.texts, text="")
        g.set_titles(col_template="{col_name}", row_template="{row_name}")

        g.fig.tight_layout()
        plt.subplots_adjust(hspace=0.2, wspace=0.1)
        g.fig.savefig(
            pdf_out,
            bbox_inches='tight',
            dpi=300
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate boxplot')
    parser.add_argument(
        'pandas_in',
        type=str,
        help='in .pandas file generated by aggregate.py',
    )
    parser.add_argument(
        'pdf_out',
        type=str,
        help='path to output pdf file',
    )
    parser.add_argument(
        '--single',
        action='store_true',
        help='single boxplot per page'
    )

    args = parser.parse_args()
    save_boxplot(args.pandas_in, args.pdf_out)
