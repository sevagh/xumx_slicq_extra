#!/usr/bin/env bash

pybin="/home/sevagh/venvs/thesis/bin/python3"

#export MUSDB_MAX_TRACKS=2
#export MUSDB_TRACK_OFFSET=15
export MUSDB_PATH=/home/sevagh/TRAINING-MUSIC/MUSDB18-HQ

#$pybin oracle_eval/ideal_mask.py --mono --eval_dir exp-octave ./config_octave.json
#$pybin oracle_eval/ideal_mask.py --mono --eval_dir exp-mel ./config_mel.json
#$pybin oracle_eval/ideal_mask.py --mono --eval_dir exp-bark ./config_bark.json
#$pybin oracle_eval/ideal_mask.py --mono --eval_dir exp-log ./config_log.json

#$pybin oracle_eval/grid_ideal_mask.py --mono
$pybin oracle_eval/grid_ideal_mask.py --mono --print-every-n=1
