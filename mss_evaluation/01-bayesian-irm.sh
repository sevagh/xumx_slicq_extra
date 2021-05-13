#!/usr/bin/env bash

set -eoxu pipefail

pybin="/home/sevagh/venvs/thesis/bin/python3"
expdir="exp-01-bayesian-irm"
scriptdir="./mss-oracle-experiments/oracle_eval"

export MUSDB_PATH=/home/sevagh/TRAINING-MUSIC/MUSDB18-HQ

mkdir -p "${expdir}"

# bayesian search with irm1 (ideal ratio mask)
#$pybin "${scriptdir}/search_best_nsgt.py" --oracle='irm1' --n-random-tracks=3 --optimization-random=20 --optimization-iter=180 --logdir "${expdir}" --fscale=vqlog #&> "${expdir}"/bayesian_out_vqlog.txt
#$pybin "${scriptdir}/search_best_nsgt.py" --oracle='irm1' --n-random-tracks=3 --optimization-random=20 --optimization-iter=180 --logdir "${expdir}" --fscale=cqlog #&> "${expdir}"/bayesian_out_cqlog.txt
#$pybin "${scriptdir}/search_best_nsgt.py" --oracle='irm1' --n-random-tracks=3 --optimization-random=20 --optimization-iter=180 --logdir "${expdir}" --fscale=mel #&> "${expdir}"/bayesian_out_mel.txt
$pybin "${scriptdir}/search_best_nsgt.py" --oracle='irm1' --n-random-tracks=3 --optimization-random=20 --optimization-iter=180 --logdir "${expdir}" --fscale=bark #&> "${expdir}"/bayesian_out_bark.txt
