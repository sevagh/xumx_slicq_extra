#!/usr/bin/env bash

pybin="/home/sevagh/venvs/thesis/bin/python3"
scriptdir="./mss-oracle-experiments/oracle_eval"

# controls - crunch into pandas + boxplot
#$pybin "${scriptdir}/aggregate.py" --out=./exp-00-controls/data.pandas ./exp-00-controls/*
#$pybin "${scriptdir}/boxplot.py" --colors-legend=control ./exp-00-controls/data.pandas ./exp-00-controls/boxplot.pdf

# pretrained model + 2 controls - crunch into pandas + boxplot
$pybin "${scriptdir}/aggregate.py" --out=./exp-04-trained-models/data.pandas ./exp-04-trained-models/*
$pybin "${scriptdir}/boxplot.py" --print-median-only ./exp-04-trained-models/data.pandas ./exp-04-trained-models/boxplot.pdf
$pybin "${scriptdir}/boxplot.py" ./exp-04-trained-models/data.pandas ./exp-04-trained-models/boxplot.pdf
