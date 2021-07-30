#!/usr/bin/env bash

pybin="/home/sevagh/venvs/thesis/bin/python3"
expdir="exp-04-trained-models"
scriptdir="./mss-oracle-experiments/oracle_eval"

#export MUSDB_MAX_TRACKS=3
export MUSDB_PATH=/home/sevagh/TRAINING-MUSIC/MUSDB18-HQ

# crunch and boxplot
$pybin "${scriptdir}/aggregate.py" --out=./"${expdir}"/data.pandas ./"${expdir}"/*
$pybin "${scriptdir}/boxplot.py" ./"${expdir}"/data.pandas ./"${expdir}"/boxplot.pdf
