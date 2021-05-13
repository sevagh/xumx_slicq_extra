#!/usr/bin/env bash

pybin="/home/sevagh/venvs/thesis/bin/python3"
expdir="exp-03-2-boxplot"
conf="config.json"
scriptdir="./mss-oracle-experiments/oracle_eval"

export MUSDB_MAX_TRACKS=3
export MUSDB_PATH=/home/sevagh/TRAINING-MUSIC/MUSDB18-HQ

# full boxplot eval
$pybin "${scriptdir}/ideal_mask.py" --eval_dir $expdir ./"${conf}"

# crunch and boxplot
$pybin "${scriptdir}/aggregate.py" --out=./"${expdir}"/data.pandas ./"${expdir}"/*
$pybin "${scriptdir}/boxplot.py" ./"${expdir}"/data.pandas ./"${expdir}"/boxplot.pdf
