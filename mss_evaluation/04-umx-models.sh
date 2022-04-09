#!/usr/bin/env bash

set -eoxu pipefail

maxmem="56" #GB
firejailcmd="firejail --noprofile --rlimit-as=${maxmem}000000000"
pybin="${firejailcmd} /home/sevagh/venvs/thesis/bin/python3"
expdir="exp-04-trained-models"
scriptdir="./mss-oracle-experiments/oracle_eval"
umxorigdir="./vendor/open-unmix-pytorch-1.0.0"
umxmodeldir="./pretrained_models/umx"
umxexpdir="exp-04-trained-models-umx-orig"

export MUSDB_PATH=/run/media/sevagh/windows-games/MDX-datasets/MUSDB18-HQ/
#export CUDA_VISIBLE_DEVICES=0
#export MUSDB_MAX_TRACKS=10

mkdir -p "${expdir}"

#$pybin "${scriptdir}/trained_models.py" --eval_dir="${expdir}/" --model=umx
#$pybin "${scriptdir}/trained_models.py" --eval_dir="${expdir}/" --model=xumx
#$pybin "${scriptdir}/trained_models.py" --eval_dir="${expdir}/" --model=slicq-wslicq
#$pybin "${scriptdir}/trained_models.py" --eval_dir="${expdir}/" --model=slicq-wstft

$pybin "${umxorigdir}/eval.py" --root="${MUSDB_PATH}" --is-wav --no-cuda --model="${umxmodeldir}" --evaldir="${umxexpdir}/"
