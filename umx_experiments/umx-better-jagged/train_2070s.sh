#!/usr/bin/env sh

musdbdir="/home/sevagh/TRAINING-MUSIC/MUSDB18-HQ/"
outdir="umx-slicq-3-2070s"

set -x

batch=128
epochs=1000
seqdur=1

declare -a targetargs=(
	"--target=drums"
#	"--target=vocals"
)

for i in "${targetargs[@]}"
do
	python scripts/train.py \
		--fscale=bark --fbins=281 --fmin=14.5 --sllen=19260 \
		--root "${musdbdir}" --is-wav --nb-workers=4 --batch-size=$batch --epochs=$epochs --random-track-mix \
		--seq-dur=$seqdur \
		$i --debug --bandwidth=16000 \
		--output "${outdir}" \
		--source-augmentations gain channelswap \
		--cuda-device=1
done
