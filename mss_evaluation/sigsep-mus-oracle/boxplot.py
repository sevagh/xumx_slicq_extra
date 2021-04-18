'''
adapted from https://github.com/sigsep/sigsep-mus-2018-analysis/blob/master/sisec-2018-paper-figures/boxplot.py
'''

import pandas
import seaborn
import matplotlib.pyplot as plt
import matplotlib
import sys

seaborn.set()

metrics = ['metrics.SDR', 'metrics.SIR', 'metrics.SAR', 'metrics.ISR']
targets = ['vocals', 'accompaniment', 'drums', 'bass', 'other']

df = pandas.read_pickle(sys.argv[1])

# aggregate methods by mean using median by track
df = df.groupby(
    ['method', 'track', 'target', 'metric']
).median().reset_index()

print(df)

# Get sorting keys (sorted by median of SDR:vocals)
df_sort_by = df[
    (df.metric == "metrics.SDR") &
    (df.target == "vocals")
]

methods_by_sdr = df_sort_by.score.groupby(
    df_sort_by.method
).median().sort_values().index.tolist()

print(methods_by_sdr)


plt.rc('text', usetex=True)
plt.rc('font', family='serif')

#matplotlib.rcParams['font.family'] = 'serif'
#matplotlib.rcParams['text.latex.unicode'] = 'True'

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

g.fig.tight_layout()
plt.subplots_adjust(hspace=0.2, wspace=0.1)
g.fig.savefig(
    "oracle_boxplot.pdf",
    bbox_inches='tight',
    dpi=300
)
