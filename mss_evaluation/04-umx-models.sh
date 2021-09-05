#!/usr/bin/env bash

set -eoxu pipefail

pybin="/home/sevagh/venvs/thesis/bin/python3"
expdir="exp-04-trained-models-for-timing"
scriptdir="./mss-oracle-experiments/oracle_eval"

export MUSDB_PATH=/home/sevagh/TRAINING-MUSIC/MUSDB18-HQ/

mkdir -p "${expdir}"

$pybin "${scriptdir}/trained_models.py" --eval_dir="${expdir}/" --model=umx
$pybin "${scriptdir}/trained_models.py" --eval_dir="${expdir}/" --model=xumx
$pybin "${scriptdir}/trained_models.py" --eval_dir="${expdir}/" --model=slicq-wslicq
$pybin "${scriptdir}/trained_models.py" --eval_dir="${expdir}/" --model=slicq-wstft
