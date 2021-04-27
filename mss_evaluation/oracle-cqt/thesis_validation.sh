#!/usr/bin/env bash

pybin="/home/sevagh/venvs/thesis/bin/python3"

#export MUSDB_MAX_TRACKS=1
#export MUSDB_TRACK_OFFSET=15
export MUSDB_PATH=/home/sevagh/TRAINING-MUSIC/MUSDB18-HQ

$pybin evaloracle/ideal_mask.py --mono --eval_dir results-octave ./config_octave.json
$pybin evaloracle/aggregate.py ./results-octave/* --out=results-octave/data.pandas
$pybin evaloracle/boxplot.py ./results-octave/data.pandas ./results-octave/boxplot.pdf

$pybin evaloracle/ideal_mask.py --mono --eval_dir results-mel ./config_mel.json
$pybin evaloracle/aggregate.py ./results-mel/* --out=results-mel/data.pandas
$pybin evaloracle/boxplot.py ./results-mel/data.pandas ./results-mel/boxplot.pdf

$pybin evaloracle/ideal_mask.py --mono --eval_dir results-bark ./config_bark.json
$pybin evaloracle/aggregate.py ./results-bark/* --out=results-bark/data.pandas
$pybin evaloracle/boxplot.py ./results-bark/data.pandas ./results-bark/boxplot.pdf

$pybin evaloracle/ideal_mask.py --mono --eval_dir results-log ./config_log.json
$pybin evaloracle/aggregate.py ./results-log/* --out=results-log/data.pandas
$pybin evaloracle/boxplot.py ./results-log/data.pandas ./results-log/boxplot.pdf
