#!/usr/bin/env bash

pybin="/home/sevagh/venvs/thesis/bin/python3"

export MUSDB_PATH=/home/sevagh/TRAINING-MUSIC/MUSDB18-HQ

$pybin oracle_eval/grid_ideal_mask.py --mono --n-random-tracks=3 --optimization-random=20 --optimization-iter=180 --logdir ./bayes-opt-logs

#$pybin oracle_eval/ideal_mask.py --mono --eval_dir exp-verif-oct ./config.json
