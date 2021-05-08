#!/usr/bin/env bash

pybin="/home/sevagh/venvs/thesis/bin/python3"
expdir="tmp2"
conf="config.json"

export MUSDB_MAX_TRACKS=4
export MUSDB_PATH=/home/sevagh/TRAINING-MUSIC/MUSDB18-HQ

# bayesian search with phasemix
#$pybin oracle_eval/search_best_nsgt.py --oracle='mpi' --n-random-tracks=3 --optimization-random=20 --optimization-iter=180 --logdir ./bayes-opt-logs
$pybin oracle_eval/search_best_nsgt.py --control --oracle='mpi' --n-random-tracks=3
$pybin oracle_eval/search_best_nsgt.py --control --oracle='irm1' --n-random-tracks=3

# full eval
#$pybin oracle_eval/ideal_mask.py --eval_dir $expdir ./"${conf}"

# crunch and boxplot
#$pybin oracle_eval/aggregate.py --out=./"${expdir}"/data.pandas ./"${expdir}"/*
#$pybin oracle_eval/boxplot.py ./"${expdir}"/data.pandas ./"${expdir}"/boxplot.pdf

# describe tf configs
#$pybin oracle_eval/describe_tf.py ./"${conf}"
