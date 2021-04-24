#!/usr/bin/env bash

exp_dir="experiment-04"
pybin="/home/sevagh/venvs/thesis/bin/python3"
kprofbin="/home/sevagh/venvs/thesis/bin/kernprof"

rm -rf ${exp_dir}

export MUSDB_MAX_TRACKS=1
export MUSDB_PATH=/home/sevagh/TRAINING-MUSIC/MUSDB18-HQ

echo "timing and profiling single track sep, bss eval"

# nfft = 2048
#time $kprofbin -l evaloracle/IRM.py --alpha=1 --eval_dir ./${exp_dir}/IRM1-STFT2048/
#$pybin -m line_profiler IRM.py.lprof
time $pybin evaloracle/IRM.py --alpha=1 --eval_dir ./${exp_dir}/IRM1-STFT2048/

exit 0
#$pybin evaloracle/IRM.py --alpha=1 --eval_dir ./${exp_dir}/IRM1-STFT2048/
#$pybin evaloracle/IRM.py --alpha=2 --eval_dir ./${exp_dir}/IRM2-STFT2048/
#$pybin evaloracle/IRM.py --binary-mask --alpha=1 --eval_dir ./${exp_dir}/IBM1-STFT2048/
#$pybin evaloracle/IRM.py --binary-mask --alpha=2 --eval_dir ./${exp_dir}/IBM2-STFT2048/

# cqt starting from 27.5hz, 96 bins per octave
time $pybin evaloracle/IRM.py --use-cqt --alpha=1 --eval_dir ./${exp_dir}/IRM1-CQT96/
#$pybin evaloracle/IRM.py --use-cqt --alpha=2 --eval_dir ./${exp_dir}/IRM2-CQT96/
#$pybin evaloracle/IRM.py --use-cqt --binary-mask --alpha=1 --eval_dir ./${exp_dir}/IBM1-CQT96/
#$pybin evaloracle/IRM.py --use-cqt --binary-mask --alpha=2 --eval_dir ./${exp_dir}/IBM2-CQT96/
#
## cqt starting from 27.5hz, 120 bins per octave
#$pybin evaloracle/IRM.py --use-cqt --alpha=1 --cqt-bins=120 --eval_dir ./${exp_dir}/IRM1-CQT120/
#$pybin evaloracle/IRM.py --use-cqt --alpha=2 --cqt-bins=120 --eval_dir ./${exp_dir}/IRM2-CQT120/
#$pybin evaloracle/IRM.py --use-cqt --binary-mask --alpha=1 --cqt-bins=120 --eval_dir ./${exp_dir}/IBM1-CQT120/
#$pybin evaloracle/IRM.py --use-cqt --binary-mask --alpha=2 --cqt-bins=120 --eval_dir ./${exp_dir}/IBM2-CQT120/
#
## cqt starting from 27.5hz, 48 bins per octave
#$pybin evaloracle/IRM.py --use-cqt --alpha=1 --cqt-bins=48 --eval_dir ./${exp_dir}/IRM1-CQT48/
#$pybin evaloracle/IRM.py --use-cqt --alpha=2 --cqt-bins=48 --eval_dir ./${exp_dir}/IRM2-CQT48/
#$pybin evaloracle/IRM.py --use-cqt --binary-mask --alpha=1 --cqt-bins=48 --eval_dir ./${exp_dir}/IBM1-CQT48/
#$pybin evaloracle/IRM.py --use-cqt --binary-mask --alpha=2 --cqt-bins=48 --eval_dir ./${exp_dir}/IBM2-CQT48/
#
## cqt starting from 27.5hz, 24 bins per octave
#$pybin evaloracle/IRM.py --use-cqt --alpha=1 --cqt-bins=24 --eval_dir ./${exp_dir}/IRM1-CQT24/
#$pybin evaloracle/IRM.py --use-cqt --alpha=2 --cqt-bins=24 --eval_dir ./${exp_dir}/IRM2-CQT24/
#$pybin evaloracle/IRM.py --use-cqt --binary-mask --alpha=1 --cqt-bins=24 --eval_dir ./${exp_dir}/IBM1-CQT24/
#$pybin evaloracle/IRM.py --use-cqt --binary-mask --alpha=2 --cqt-bins=24 --eval_dir ./${exp_dir}/IBM2-CQT24/
#
## cqt starting from 27.5hz, 12 bins per octave
#$pybin evaloracle/IRM.py --use-cqt --alpha=1 --cqt-bins=12 --eval_dir ./${exp_dir}/IRM1-CQT12/
#$pybin evaloracle/IRM.py --use-cqt --alpha=2 --cqt-bins=12 --eval_dir ./${exp_dir}/IRM2-CQT12/
#$pybin evaloracle/IRM.py --use-cqt --binary-mask --alpha=1 --cqt-bins=12 --eval_dir ./${exp_dir}/IBM1-CQT12/
#$pybin evaloracle/IRM.py --use-cqt --binary-mask --alpha=2 --cqt-bins=12 --eval_dir ./${exp_dir}/IBM2-CQT12/

$pybin evaloracle/aggregate.py ./${exp_dir}/* --out=${exp_dir}/sisecc.pandas
$pybin evaloracle/boxplot.py ./${exp_dir}/sisecc.pandas ./${exp_dir}/oracle_boxplot.pdf
