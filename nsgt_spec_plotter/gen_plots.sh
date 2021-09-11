#!/usr/bin/env bash

set -eoxu pipefail

pybin="/home/sevagh/venvs/thesis/bin/python3"
scriptdir="./nsgt/examples"
gspipath="../matlab/vendor/ltfat/signals/gspi.wav"

${pybin} ${scriptdir}/spectrogram.py --scale=bark --fmin=32.9 --fmax=22050 --bins=262 --plot --cmap inferno "${gspipath}" --fontsize=40
