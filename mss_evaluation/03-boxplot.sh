#!/usr/bin/env bash

pybin="/home/sevagh/venvs/thesis/bin/python3"
scriptdir="./mss-oracle-experiments/oracle_eval"

export MUSDB_PATH=/home/sevagh/TRAINING-MUSIC/MUSDB18-HQ

# controls - crunch into pandas + boxplot
#$pybin "${scriptdir}/aggregate.py" --out=./exp-00-controls/data.pandas ./exp-00-controls/*
#$pybin "${scriptdir}/boxplot.py" --colors-legend=control ./exp-00-controls/data.pandas ./exp-00-controls/boxplot.pdf

# pretrained model + 2 controls - crunch into pandas + boxplot
#$pybin "${scriptdir}/aggregate.py" --out=./exp-04-trained-models-with-controls/data.pandas ./exp-04-trained-models-with-controls/*
$pybin "${scriptdir}/boxplot.py" --print-median-only --colors-legend=pretrained ./exp-04-trained-models-with-controls/data.pandas ./exp-04-trained-models-with-controls/boxplot.pdf
