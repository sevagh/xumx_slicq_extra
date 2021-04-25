#!/usr/bin/env bash

exp_dir="experiment-04"
pybin="/home/sevagh/venvs/thesis/bin/python3"

rm -rf ${exp_dir}

export MUSDB_MAX_TRACKS=1
export MUSDB_PATH=/home/sevagh/TRAINING-MUSIC/MUSDB18-HQ

time $pybin evaloracle/ideal_mask.py --eval_dir ./${exp_dir}/ ./config.json

$pybin evaloracle/aggregate.py ./${exp_dir}/* --out=${exp_dir}/sisecc.pandas
$pybin evaloracle/boxplot.py ./${exp_dir}/sisecc.pandas ./${exp_dir}/oracle_boxplot.pdf
