#!/usr/bin/env bash

pybin="/home/sevagh/venvs/thesis/bin/python3"
expdir="tmp2"
conf="final_config.json"

#export MUSDB_MAX_TRACKS=1
export MUSDB_PATH=/home/sevagh/TRAINING-MUSIC/MUSDB18-HQ

# bayesian search with phasemix
$pybin oracle_eval/grid_ideal_mask.py --phasemix --mono --n-random-tracks=3 --optimization-random=20 --optimization-iter=180 --logdir ./bayes-opt-logs-phasemix

# full eval
#$pybin oracle_eval/ideal_mask.py --mono --eval_dir $expdir ./"${conf}"

# crunch and boxplot
#$pybin oracle_eval/aggregate.py --out=./"${expdir}"/data.pandas ./"${expdir}"/irm1-*
#$pybin oracle_eval/boxplot.py ./"${expdir}"/data.pandas ./"${expdir}"/boxplot.pdf

# describe tf configs
#$pybin oracle_eval/describe_tf.py ./"${conf}"

# bayesian control
#$pybin oracle_eval/grid_ideal_mask.py --mono --n-random-tracks=3 --control
#$pybin oracle_eval/grid_ideal_mask.py --mono --n-random-tracks=3 --control --phasemix
