#!/usr/bin/env bash

pybin="/home/sevagh/venvs/thesis/bin/python3"

$pybin oracle_eval/aggregate.py ./exp-octave/* --out=exp-octave/data.pandas
$pybin oracle_eval/boxplot.py ./exp-octave/data.pandas ./exp-octave/boxplot.pdf

#$pybin oracle_eval/aggregate.py ./exp-mel/* --out=exp-mel/data.pandas
#$pybin oracle_eval/boxplot.py ./exp-mel/data.pandas ./exp-mel/boxplot.pdf
#
#$pybin oracle_eval/aggregate.py ./exp-bark/* --out=exp-bark/data.pandas
#$pybin oracle_eval/boxplot.py ./exp-bark/data.pandas ./exp-bark/boxplot.pdf
#
#$pybin oracle_eval/aggregate.py ./exp-log/* --out=exp-log/data.pandas
#$pybin oracle_eval/boxplot.py ./exp-log/data.pandas ./exp-log/boxplot.pdf
