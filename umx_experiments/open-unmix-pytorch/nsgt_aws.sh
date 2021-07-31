#!/usr/bin/env sh

musdbdir="/home/sevagh/TRAINING-MUSIC/MUSDB18-HQ/"
outdir="umx-nsgt-2"

batch=32
epochs=1000
workers=4

for target in drums other vocals bass; do
	python scripts/train.py --root "${musdbdir}" --target=$target --epochs=$epochs --is-wav --nb-workers=$workers --batch-size=$batch --output "${outdir}" --seq-dur=1 --debug
done

#python -m openunmix.evaluate --root "${musdbdir}" --is-wav --outdir=./ests-$outdir --evaldir=./results-$outdir --model="${outdir}" --no-cuda
