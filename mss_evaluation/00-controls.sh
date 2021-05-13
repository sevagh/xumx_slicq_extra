#!/usr/bin/env bash

set -eoxu pipefail

pybin="/home/sevagh/venvs/thesis/bin/python3"
expdir="exp-00-controls"
scriptdir="./mss-oracle-experiments/oracle_eval"

export MUSDB_PATH=/home/sevagh/TRAINING-MUSIC/MUSDB18-HQ

mkdir -p "${expdir}"

# evaluate control stfts with irm1
$pybin "${scriptdir}/search_best_nsgt.py" --control --oracle='irm1' --n-random-tracks=3 #&> "${expdir}"/controls_out_irm1.txt

# evaluate control stfts with mpi
$pybin "${scriptdir}/search_best_nsgt.py" --control --oracle='mpi' --n-random-tracks=3 #&> "${expdir}"/controls_out_mpi.txt
