#!/usr/bin/env sh

musdbdir="/home/sevagh/TRAINING-MUSIC/MUSDB18-HQ/"

evaldir="umx-ep-20210610"

echo "evaluating $evaldir"

# musiphery evaluation
python -m openunmix.evaluate --root "${musdbdir}" --is-wav --outdir=./ests-$evaldir --evaldir=./results-$evaldir --model="${evaldir}" --no-cuda
