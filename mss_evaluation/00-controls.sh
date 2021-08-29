#!/usr/bin/env bash

set -eoxu pipefail

pybin="/home/sevagh/venvs/thesis/bin/python3"
expdir="exp-00-controls"
scriptdir="./mss-oracle-experiments/oracle_eval"

export MUSDB_PATH=/home/sevagh/TRAINING-MUSIC/MUSDB18-HQ

mkdir -p "${expdir}"

argument_list="--oracle=irm1
--oracle=irm2
--oracle=ibm1
--oracle=ibm2
--oracle=mpi"

# limit to 2
echo "${argument_list}" | parallel --jobs=2 --colsep ' ' --ungroup $pybin "${scriptdir}/search_best_nsgt.py" --control --eval-dir="${expdir}" --cuda-device={#}
