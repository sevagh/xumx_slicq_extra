#!/usr/bin/env sh

musdbdir="/home/ubuntu/musdb18hq/"
outdir="umx-nsgt-1"

set -x

batch=32
epochs=1000
workers=4
#extraflag="--debug"
extraflag=""

for target in vocals other drums bass; do
	python scripts/train.py --root "${musdbdir}"  --target=$target --epochs=$epochs --is-wav --nb-workers=$workers --batch-size=$batch --output "${outdir}"
done

python -m openunmix.evaluate --root "${musdbdir}" --is-wav --outdir=./ests-$outdir --evaldir=./results-$outdir --model="${outdir}" --no-cuda
