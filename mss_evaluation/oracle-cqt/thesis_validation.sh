#!/usr/bin/env bash

exp_dir="experiment-04"
pybin="/home/sevagh/venvs/thesis/bin/python3"

rm -rf ${exp_dir}

export MUSDB_MAX_TRACKS=1
export MUSDB_PATH=/home/sevagh/TRAINING-MUSIC/MUSDB18-HQ

# cqt starting from 27.5hz, 96 bins per octave
echo "debug cq-nsgt perf"
pybin="/home/sevagh/venvs/nsgt-accel/bin/python3"
time $pybin evaloracle/IRM.py --eval_dir ./${exp_dir}/ ./config.json

exit 0

$pybin evaloracle/aggregate.py ./${exp_dir}/* --out=${exp_dir}/sisecc.pandas
$pybin evaloracle/boxplot.py ./${exp_dir}/sisecc.pandas ./${exp_dir}/oracle_boxplot.pdf
