#!/usr/bin/env bash

set -eoxu pipefail

pybin="/home/sevagh/venvs/thesis/bin/python3"
scriptdir="./nsgt/examples"
gspipath="../matlab/vendor/ltfat/signals/gspi.wav"

# psychoacoustic scales

${pybin} ${scriptdir}/spectrogram.py --scale=mel --fmin=20 --fmax=22050 --bins=500 --plot --cmap inferno "${gspipath}" --fontsize=34 --output="../latex/images-gspi/gspi_nsgt_mel_500.png" --nonsliced

${pybin} ${scriptdir}/spectrogram.py --scale=mel --fmin=20 --fmax=22050 --bins=100 --plot --cmap inferno "${gspipath}" --fontsize=34 --output="../latex/images-gspi/gspi_nsgt_mel_100.png" --nonsliced

${pybin} ${scriptdir}/spectrogram.py --scale=bark --fmin=20 --fmax=22050 --bins=100 --plot --cmap inferno "${gspipath}" --fontsize=34 --output="../latex/images-gspi/gspi_nsgt_bark_100.png" --nonsliced

${pybin} ${scriptdir}/spectrogram.py --scale=bark --fmin=20 --fmax=22050 --bins=500 --plot --cmap inferno "${gspipath}" --fontsize=34 --output="../latex/images-gspi/gspi_nsgt_bark_500.png" --nonsliced

# constant-q and variable-q

${pybin} ${scriptdir}/spectrogram.py --scale=cqlog --fmin=20 --fmax=22050 --bins=100 --plot --cmap inferno "${gspipath}" --fontsize=34 --output="../latex/images-gspi/gspi_nsgt_cqlog_100.png" --nonsliced

${pybin} ${scriptdir}/spectrogram.py --scale=cqlog --fmin=20 --fmax=22050 --bins=500 --plot --cmap inferno "${gspipath}" --fontsize=34 --output="../latex/images-gspi/gspi_nsgt_cqlog_500.png" --nonsliced

${pybin} ${scriptdir}/spectrogram.py --scale=vqlog --gamma=15 --fmin=20 --fmax=22050 --bins=100 --plot --cmap inferno "${gspipath}" --fontsize=34 --output="../latex/images-gspi/gspi_nsgt_vqlog_100.png" --nonsliced

${pybin} ${scriptdir}/spectrogram.py --scale=vqlog --gamma=15 --fmin=20 --fmax=22050 --bins=500 --plot --cmap inferno "${gspipath}" --fontsize=34 --output="../latex/images-gspi/gspi_nsgt_vqlog_500.png" --nonsliced

# misc, overlaps, etc.

${pybin} ${scriptdir}/spectrogram.py --scale=mel --fmin=20 --fmax=22050 --bins=96 --plot --cmap inferno "${gspipath}" --fontsize=34 --nonsliced --output="../latex/images-gspi/gspi_nsgt_mel_nooverlap.png"

${pybin} ${scriptdir}/spectrogram.py --scale=mel --fmin=20 --fmax=22050 --bins=96 --plot --cmap inferno "${gspipath}" --fontsize=34 --output="../latex/images-gspi/gspi_nsgt_mel_perfect_slice.png"

${pybin} ${scriptdir}/spectrogram.py --scale=bark --fmin=20 --fmax=22050 --bins=84 --plot --cmap inferno "${gspipath}" --fontsize=34 --flatten --output="../latex/images-gspi/gspi_overlap_flatten.png"

${pybin} ${scriptdir}/spectrogram.py --scale=bark --fmin=20 --fmax=22050 --bins=84 --plot --cmap inferno "${gspipath}" --fontsize=34 --output="../latex/images-gspi/gspi_overlap_proper.png"

${pybin} ${scriptdir}/spectrogram.py --scale=cqlog --fmin=129.7 --fmax=22050 --bins=142 --plot --cmap inferno "${gspipath}" --fontsize=34 --output="../latex/images-gspi/slicqt_bad.png"

${pybin} ${scriptdir}/spectrogram.py --scale=bark --fmin=32.9 --fmax=22050 --bins=262 --plot --cmap inferno "${gspipath}" --fontsize=34 --output="../latex/images-gspi/slicqt_good.png"
