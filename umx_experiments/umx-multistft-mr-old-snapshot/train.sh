#!/usr/bin/env sh

musdbdir="/home/sevagh/TRAINING-MUSIC/MUSDB18-HQ/"

epochs=1000
workers=4
batch_size=32
seq_dur=6

#for target in drums vocals bass other;
for target in drums;
do
	python scripts/train.py \
		--root "${musdbdir}" --is-wav --nb-workers=$workers --batch-size=$batch_size --epochs=$epochs \
		--target="$target" \
		--seq-dur=$seq_dur --valid-seq-dur=$seq_dur --valid-batch-size=$batch_size \
		--output="umx-mr-5" --debug
done


#python -m openunmix.evaluate --root "${musdbdir}" --is-wav --outdir=./ests-$evaldir --evaldir=./results-$evaldir --model="${evaldir}" --no-cuda
