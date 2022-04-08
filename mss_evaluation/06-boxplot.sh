#!/usr/bin/env bash

pybin="/home/sevagh/venvs/thesis/bin/python3"
scriptdir="./mss-oracle-experiments/oracle_eval"
resultsdir="./eval-20220408"

# pretrained models - crunch into pandas + boxplot
$pybin "${scriptdir}/aggregate.py" --out="${resultsdir}/data.pandas" "${resultsdir}/*"
$pybin "${scriptdir}/boxplot.py" --print-median-only "${resultsdir}/data.pandas" "${resultsdir}/boxplot.pdf"
$pybin "${scriptdir}/boxplot.py" "${resultsdir}/data.pandas" "${resultsdir}/boxplot.pdf" --colors-legend=pretrained
