#!/usr/bin/env sh

#musdbdir="/home/sevagh/TRAINING-MUSIC/MUSDB18-HQ/"
datadir="/home/sevagh/TRAINING-MUSIC/MUSIPHERY/"

epochs=1000
workers=4
batch=32

python scripts/train.py \
	--root="${datadir}" --dataset='trackfolder_fix' \
	--target-file="drums.wav" \
	--interferer-files="bass.wav,vocals.wav,other.wav" \
	--random-track-mix \
	--batch-size=$batch --epochs=$epochs --nb-workers=$workers \
	--target="drums" \
	--output="umx-p-1" #--debug

python scripts/train.py \
	--root="${datadir}" --dataset='trackfolder_fix' \
	--target-file="bass.wav" \
	--interferer-files="drums.wav,vocals.wav,other.wav" \
	--random-track-mix \
	--batch-size=$batch --epochs=$epochs --nb-workers=$workers \
	--target="bass" \
	--output="umx-p-1" #--debug

python scripts/train.py \
	--root="${datadir}" --dataset='trackfolder_fix' \
	--target-file="other.wav" \
	--interferer-files="drums.wav,vocals.wav,bass.wav" \
	--random-track-mix \
	--batch-size=$batch --epochs=$epochs --nb-workers=$workers \
	--target="other" \
	--output="umx-p-1" #--debug

python scripts/train.py \
	--root="${datadir}" --dataset='trackfolder_fix' \
	--target-file="vocals.wav" \
	--interferer-files="drums.wav,other.wav,bass.wav" \
	--random-track-mix \
	--batch-size=$batch --epochs=$epochs --nb-workers=$workers \
	--target="vocals" \
	--output="umx-p-1" #--debug

#python -m openunmix.evaluate --root "${musdbdir}" --is-wav --outdir=./ests-$evaldir --evaldir=./results-$evaldir --model="${evaldir}" --no-cuda
