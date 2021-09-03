#!/usr/bin/env bash

set -eoxu pipefail

pybin="/home/sevagh/venvs/thesis/bin/python3"
expdir="exp-04-trained-models-with-controls/"
scriptdir="./mss-oracle-experiments/oracle_eval"

export MUSDB_PATH=/home/sevagh/TRAINING-MUSIC/MUSDB18-HQ
#export MUSDB_MAX_TRACKS=1

mkdir -p "${expdir}"

params=(
#"--oracle=irm1 --control-window-sizes=256"
#"--oracle=mpi --control-window-sizes=256"
#"--oracle=irm1 --control-window-sizes=512"
#"--oracle=mpi --control-window-sizes=512"
#"--oracle=irm1 --control-window-sizes=1024"
#"--oracle=mpi --control-window-sizes=1024"
#"--oracle=irm1 --control-window-sizes=2048"
#"--oracle=mpi --control-window-sizes=2048"
"--oracle=irm1 --control-window-sizes=4096 --control"
"--oracle=mpi --control-window-sizes=4096 --control"
#"--oracle=irm1 --control-window-sizes=8192"
#"--oracle=mpi --control-window-sizes=8192"
#"--oracle=irm1 --control-window-sizes=16384"
#"--oracle=mpi --control-window-sizes=16384"
"--oracle=irm1 --fixed-slicqt-param=bark,262,32.9 --fixed-slicqt"
"--oracle=mpi --fixed-slicqt-param=bark,262,32.9 --fixed-slicqt"
#"--oracle=irm1 --fixed-slicqt-param=cqlog,142,129.7"
#"--oracle=mpi --fixed-slicqt-param=cqlog,142,129.7"
)

# limit to 2
parallel --jobs=1 --colsep=' ' --ungroup $pybin "${scriptdir}/search_best_nsgt.py" --eval-dir="${expdir}" --cuda-device={#} {} ::: "${params[@]}" 

#time $pybin -m kernprof -l -v "${scriptdir}/search_best_nsgt.py" --fixed-slicqt --eval-dir="${expdir}" --cuda-device=0 --oracle=irm1 --control-window-sizes=4096 --control
