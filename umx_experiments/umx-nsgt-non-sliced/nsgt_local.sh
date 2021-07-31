#!/usr/bin/env sh

musdbdir="/home/sevagh/TRAINING-MUSIC/MUSDB18-HQ/"
outdir="umx-nsgt-linear"

batch=12
epochs=4
workers=4
seqdur=6

#for target in vocals drums other bass;
#for target in drums other bass;
#do
#	python scripts/train.py \
#		--root "${musdbdir}" --is-wav --nb-workers=$workers --batch-size=$batch --epochs=$epochs \
#		--seq-dur=$seqdur --stats-seq-dur=$seqdur --valid-seq-dur=$seqdur --valid-samples-per-track=64 --valid-batch-size=4 \
#		--target="$target" \
#		--rnn-layers=3 \
#		--output "${outdir}" --debug #--model="${outdir}"
#done

		#--bandwidth=16000 \

evaldir=$outdir

python -m openunmix.evaluate --root "${musdbdir}" --is-wav --outdir=./ests-$evaldir --evaldir=./results-$evaldir --model="${evaldir}" #--no-cuda
