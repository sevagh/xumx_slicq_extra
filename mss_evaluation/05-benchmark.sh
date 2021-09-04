#!/usr/bin/env bash

set -eoxu pipefail

pybin="/home/sevagh/venvs/thesis/bin/python3"
scriptdir="./mss-oracle-experiments/oracle_eval"

export MUSDB_PATH=/home/sevagh/TRAINING-MUSIC/MUSDB18-HQ
#export MUSDB_MAX_TRACKS=1

#$pybin "${scriptdir}/benchmark_cupy_eval.py" --cuda-device=0 --bench-iter=10 --split=valid
#$pybin "${scriptdir}/benchmark_cupy_eval.py" --cuda-device=0 --bench-iter=10 --split=valid --disable-cupy
$pybin "${scriptdir}/benchmark_cupy_eval.py" --cuda-device=1 --bench-iter=10 --split=valid
