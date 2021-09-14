#!/usr/bin/env bash

set -eoxu pipefail

pybin="/home/sevagh/venvs/thesis/bin/python3"
scriptdir="./nsgt/examples"
gspipath="../matlab/vendor/ltfat/signals/gspi.wav"

# psychoacoustic scales

#${pybin} ${scriptdir}/spectrogram.py --scale=mel --fmin=20 --fmax=22050 --bins=500 --plot --cmap inferno "${gspipath}" --fontsize=34 --output="../latex/images-gspi/gspi_nsgt_mel_500.png"
#
#${pybin} ${scriptdir}/spectrogram.py --scale=mel --fmin=20 --fmax=22050 --bins=100 --plot --cmap inferno "${gspipath}" --fontsize=34 --output="../latex/images-gspi/gspi_nsgt_mel_100.png"
#
#${pybin} ${scriptdir}/spectrogram.py --scale=bark --fmin=20 --fmax=22050 --bins=100 --plot --cmap inferno "${gspipath}" --fontsize=34 --output="../latex/images-gspi/gspi_nsgt_bark_100.png"
#
#${pybin} ${scriptdir}/spectrogram.py --scale=bark --fmin=20 --fmax=22050 --bins=500 --plot --cmap inferno "${gspipath}" --fontsize=34 --output="../latex/images-gspi/gspi_nsgt_bark_500.png"

# constant-q and variable-q

#${pybin} ${scriptdir}/spectrogram.py --scale=cqlog --fmin=20 --fmax=22050 --bins=100 --plot --cmap inferno "${gspipath}" --fontsize=34 --output="../latex/images-gspi/gspi_nsgt_cqlog_100.png"
#
#${pybin} ${scriptdir}/spectrogram.py --scale=cqlog --fmin=20 --fmax=22050 --bins=500 --plot --cmap inferno "${gspipath}" --fontsize=34 --output="../latex/images-gspi/gspi_nsgt_cqlog_500.png"
#
#${pybin} ${scriptdir}/spectrogram.py --scale=vqlog --gamma=15 --fmin=20 --fmax=22050 --bins=100 --plot --cmap inferno "${gspipath}" --fontsize=34 --output="../latex/images-gspi/gspi_nsgt_vqlog_100.png"
#
#${pybin} ${scriptdir}/spectrogram.py --scale=vqlog --gamma=15 --fmin=20 --fmax=22050 --bins=500 --plot --cmap inferno "${gspipath}" --fontsize=34 --output="../latex/images-gspi/gspi_nsgt_vqlog_500.png"


