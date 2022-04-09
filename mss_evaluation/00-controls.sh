#!/usr/bin/env bash

set -eoxu pipefail

CUDA_VISIBLE_DEVICES=1
pybin="/home/sevagh/venvs/thesis/bin/python3"
scriptdir="./mss-oracle-experiments/oracle_eval"
expdir="exp-00-controls"

export MUSDB_PATH=/run/media/sevagh/windows-games/MDX-datasets/MUSDB18-HQ/

mkdir -p "${expdir}"

control_params=(
#"--oracle=irm1 --control-window-sizes=256 --control"
#"--oracle=mpi --control-window-sizes=256 --control"
#"--oracle=irm1 --control-window-sizes=512 --control"
#"--oracle=mpi --control-window-sizes=512 --control"
#"--oracle=irm1 --control-window-sizes=1024 --control"
#"--oracle=mpi --control-window-sizes=1024 --control"
#"--oracle=irm1 --control-window-sizes=2048 --control"
#"--oracle=mpi --control-window-sizes=2048 --control"
#"--oracle=irm1 --control-window-sizes=4096 --control"
"--oracle=mpi --control-window-sizes=4096 --control"
#"--oracle=irm1 --control-window-sizes=8192 --control"
#"--oracle=mpi --control-window-sizes=8192 --control"
#"--oracle=irm1 --control-window-sizes=16384 --control"
#"--oracle=mpi --control-window-sizes=16384 --control"
#"--oracle=irm1 --fixed-slicqt-param=bark,262,32.9 --fixed-slicqt"
"--oracle=mpi --fixed-slicqt-param=bark,262,32.9 --fixed-slicqt"
#"--oracle=irm1 --fixed-slicqt-param=cqlog,142,129.7 --fixed-slicqt"
#"--oracle=mpi --fixed-slicqt-param=cqlog,142,129.7 --fixed-slicqt"
)

#eval_control_params=(
#"--oracle=irm1 --control-window-sizes=4096 --control"
#"--oracle=mpi --control-window-sizes=4096 --control"
#"--oracle=irm1 --fixed-slicqt-param=bark,262,32.9 --fixed-slicqt"
#"--oracle=mpi --fixed-slicqt-param=bark,262,32.9 --fixed-slicqt"
#)

parallel --jobs=2 --colsep=' ' --ungroup $pybin "${scriptdir}/search_best_nsgt.py" --split='test' --eval-dir=$expdir {} ::: "${control_params[@]}" 

#parallel --jobs=1 --colsep=' ' --ungroup $pybin "${scriptdir}/search_best_nsgt.py" --split='test' --eval-dir=exp-04-trained-models-with-controls --cuda-device={#} {} ::: "${eval_control_params[@]}" 

#time $pybin -m kernprof -l -v "${scriptdir}/search_best_nsgt.py" --fixed-slicqt --eval-dir="${expdir}" --cuda-device=0 --oracle=irm1 --control-window-sizes=4096 --control
