#!/usr/bin/env sh

musdbdir="/home/sevagh/TRAINING-MUSIC/MUSDB18-HQ/"
outdir="umx-nsgt-3"

batch=16
epochs=1000
workers=4

for target in drums; do
	python scripts/train.py --root "${musdbdir}" --target=$target --epochs=$epochs --is-wav --nb-workers=$workers --batch-size=$batch --output "${outdir}" --seq-dur=2 --stats-seq-dur=180 --valid-seq-dur=2 --valid-samples-per-track=12 --debug
done

#evaldir="$outdir"
#
#python -m openunmix.evaluate --root "${musdbdir}" --is-wav --outdir=./ests-$evaldir --evaldir=./results-$evaldir --model="${evaldir}" --no-cuda
