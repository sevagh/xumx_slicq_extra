#!/usr/bin/env sh

musdbdir="/home/sevagh/TRAINING-MUSIC/MUSDB18-HQ/"
outdir="umx-nsgt-cdae"

batch=64
epochs=100
workers=4

#for target in drums vocals other bass;
for target in drums;
do
	python scripts/train.py \
		--root "${musdbdir}" --is-wav --nb-workers=$workers --batch-size=$batch --epochs=$epochs \
		--valid-samples-per-track=64 --valid-batch-size=32 \
		--target="$target" \
		--output "${outdir}" #--debug # --model="${outdir}"
done

#evaldir=$outdir
#
#python -m openunmix.evaluate --root "${musdbdir}" --is-wav --outdir=./ests-$evaldir --evaldir=./results-$evaldir --model="${evaldir}" #--no-cuda
