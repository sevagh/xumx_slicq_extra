#!/usr/bin/env sh

musdbdir="/home/sevagh/TRAINING-MUSIC/MUSDB18-HQ/"
outdir="umx-nsgt-8"

batch=32
epochs=100
workers=4

seqdur=0.1

#for target in drums vocals bass other; do
for target in drums; do
	python scripts/train.py --root "${musdbdir}" --target=$target --epochs=$epochs --is-wav --nb-workers=$workers --batch-size=$batch --output "${outdir}" --seq-dur=$seqdur --stats-seq-dur=$seqdur --valid-seq-dur=$seqdur --valid-samples-per-track=64 --valid-batch-size=4 --hidden-size=512 --rnn-layers=3 --debug
done

#python -m kernprof -l -v scripts/train.py --root "${musdbdir}" --target=$target --epochs=$epochs --is-wav --nb-workers=$workers --batch-size=$batch --output "${outdir}" --seq-dur=3 --stats-seq-dur=6 --valid-seq-dur=6 --valid-samples-per-track=64 --valid-batch-size=4 --hidden-size=512 --rnn-layers=3 --debug

#evaldir="$outdir"
#
#python -m openunmix.evaluate --root "${musdbdir}" --is-wav --outdir=./ests-$evaldir --evaldir=./results-$evaldir --model="${evaldir}" --no-cuda
