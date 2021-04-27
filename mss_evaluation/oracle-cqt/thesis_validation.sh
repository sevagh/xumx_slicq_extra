#!/usr/bin/env bash

pybin="/home/sevagh/venvs/thesis/bin/python3"

#export MUSDB_MAX_TRACKS=1
#export MUSDB_TRACK_OFFSET=15
export MUSDB_PATH=/home/sevagh/TRAINING-MUSIC/MUSDB18-HQ

$pybin oracle_eval/ideal_mask.py --mono --eval_dir exp-octave ./config_octave.json
$pybin oracle_eval/aggregate.py ./exp-octave/* --out=exp-octave/data.pandas
$pybin oracle_eval/boxplot.py ./exp-octave/data.pandas ./exp-octave/boxplot.pdf

$pybin oracle_eval/ideal_mask.py --mono --eval_dir exp-mel ./config_mel.json
$pybin oracle_eval/aggregate.py ./exp-mel/* --out=exp-mel/data.pandas
$pybin oracle_eval/boxplot.py ./exp-mel/data.pandas ./exp-mel/boxplot.pdf

$pybin oracle_eval/ideal_mask.py --mono --eval_dir exp-bark ./config_bark.json
$pybin oracle_eval/aggregate.py ./exp-bark/* --out=exp-bark/data.pandas
$pybin oracle_eval/boxplot.py ./exp-bark/data.pandas ./exp-bark/boxplot.pdf

$pybin oracle_eval/ideal_mask.py --mono --eval_dir exp-log ./config_log.json
$pybin oracle_eval/aggregate.py ./exp-log/* --out=exp-log/data.pandas
$pybin oracle_eval/boxplot.py ./exp-log/data.pandas ./exp-log/boxplot.pdf
