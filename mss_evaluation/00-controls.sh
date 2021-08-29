#!/usr/bin/env bash

set -eoxu pipefail

pybin="/home/sevagh/venvs/thesis/bin/python3"
expdir="exp-00-controls"
scriptdir="./mss-oracle-experiments/oracle_eval"

export MUSDB_PATH=/home/sevagh/TRAINING-MUSIC/MUSDB18-HQ

mkdir -p "${expdir}"

params=(
"--oracle=irm1 --control-window-sizes=256"
"--oracle=irm2 --control-window-sizes=256"
"--oracle=ibm1 --control-window-sizes=256"
"--oracle=ibm2 --control-window-sizes=256"
"--oracle=mpi --control-window-sizes=256"
"--oracle=irm1 --control-window-sizes=512"
"--oracle=irm2 --control-window-sizes=512"
"--oracle=ibm1 --control-window-sizes=512"
"--oracle=ibm2 --control-window-sizes=512"
"--oracle=mpi --control-window-sizes=512"
"--oracle=irm1 --control-window-sizes=1024"
"--oracle=irm2 --control-window-sizes=1024"
"--oracle=ibm1 --control-window-sizes=1024"
"--oracle=ibm2 --control-window-sizes=1024"
"--oracle=mpi --control-window-sizes=1024"
"--oracle=irm1 --control-window-sizes=2048"
"--oracle=irm2 --control-window-sizes=2048"
"--oracle=ibm1 --control-window-sizes=2048"
"--oracle=ibm2 --control-window-sizes=2048"
"--oracle=mpi --control-window-sizes=2048"
"--oracle=irm1 --control-window-sizes=4096"
"--oracle=irm2 --control-window-sizes=4096"
"--oracle=ibm1 --control-window-sizes=4096"
"--oracle=ibm2 --control-window-sizes=4096"
"--oracle=mpi --control-window-sizes=4096"
"--oracle=irm1 --control-window-sizes=8192"
"--oracle=irm2 --control-window-sizes=8192"
"--oracle=ibm1 --control-window-sizes=8192"
"--oracle=ibm2 --control-window-sizes=8192"
"--oracle=mpi --control-window-sizes=8192"
"--oracle=irm1 --control-window-sizes=16384"
"--oracle=irm2 --control-window-sizes=16384"
"--oracle=ibm1 --control-window-sizes=16384"
"--oracle=ibm2 --control-window-sizes=16384"
"--oracle=mpi --control-window-sizes=16384"
)

# limit to 2
parallel --jobs=2 --colsep=' ' --ungroup $pybin "${scriptdir}/search_best_nsgt.py" --control --eval-dir="${expdir}" --cuda-device={#} {} ::: "${params[@]}" 
