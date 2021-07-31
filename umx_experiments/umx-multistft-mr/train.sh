#!/usr/bin/env sh

musdbdir="/home/sevagh/TRAINING-MUSIC/MUSDB18-HQ/"

epochs=1000
workers=8
batch_size=16
seq_dur=1

for target in drums vocals bass other;
do
	python scripts/train.py \
		--root "${musdbdir}" --is-wav --nb-workers=$workers --batch-size=$batch_size --epochs=$epochs \
		--target="$target" \
		--seq-dur=$seq_dur \
		--output="umx-mr-1" --debug \
		--source-augmentations gain channelswap
done

#python -m openunmix.evaluate --root "${musdbdir}" --is-wav --outdir=./ests-$evaldir --evaldir=./results-$evaldir --model="${evaldir}" --no-cuda
