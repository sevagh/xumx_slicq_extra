#!/usr/bin/env sh

musdbdir="/home/sevagh/TRAINING-MUSIC/MUSDB18-HQ/"
musdbdebug="/home/sevagh/musdbdebug"
outdir="umx-slicq-1"

batch=1
epochs=20
workers=4
seqdur=6

#declare -a targetargs=(
#	"--target=vocals --fscale=mel --fbins=116 --fmin=37.7 --sllen=8024"
#	"--target=drums --fscale=mel --fbins=104 --fmin=49.3 --sllen=7108"
#	"--target=other --fscale=bark --fbins=64 --fmin=90.0 --sllen=4416"
#	"--target=bass --fscale=bark --fbins=105 --fmin=25.4 --sllen=7180"
#)

declare -a targetargs=(
	"--target=vocals --fscale=bark --fbins=30 --fmin=38.5 --sllen=2012"
)

for i in "${targetargs[@]}"
do
	python scripts/train.py \
		--root "${musdbdebug}" --is-wav --nb-workers=$workers --batch-size=$batch --epochs=$epochs \
		--seq-dur=$seqdur --samples-per-track=1 --fixed-start=11 \
		$i --skip-statistics \
		--output "${outdir}" --patience=1000
done
