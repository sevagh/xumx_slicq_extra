#!/usr/bin/env bash

pybin="/home/sevagh/venvs/thesis/bin/python3"

export MUSDB_MAX_TRACKS=2
export MUSDB_PATH=/home/sevagh/TRAINING-MUSIC/MUSDB18-HQ

$pybin oracle_eval/grid_ideal_mask.py --mono

#$pybin oracle_eval/ideal_mask.py --mono --eval_dir exp-verif-oct ./config.json
