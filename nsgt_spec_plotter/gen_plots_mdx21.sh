#!/usr/bin/env bash

set -eoxu pipefail

pybin="/home/sevagh/venvs/thesis/bin/python3"
scriptdir="./nsgt/examples"
gspipath="../matlab/vendor/ltfat/signals/gspi.wav"

${pybin} ${scriptdir}/spectrogram.py --scale=Bark --fmin=32.9 --fmax=22050 --bins=262 --plot --cmap inferno "${gspipath}" --fontsize=34 --output="../latex/mdx-submissions21/slicq.png"

${pybin} stft_spectrogram.py --plot --cmap inferno "${gspipath}" --fontsize=34 --output="../latex/mdx-submissions21/stft.png"
${pybin} stft_spectrogram.py --plot --window=1024 --overlap=256 --cmap inferno "${gspipath}" --fontsize=34 --output="../latex/mdx-submissions21/stft_small.png"

${pybin} stft_spectrogram.py --plot --window=16384 --overlap=4096 --cmap inferno "${gspipath}" --fontsize=34 --output="../latex/mdx-submissions21/stft_big.png"
