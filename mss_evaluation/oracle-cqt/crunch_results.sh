#!/usr/bin/env bash

pybin="/home/sevagh/venvs/thesis/bin/python3"

$pybin oracle_eval/aggregate.py --out=exp-octave/data_20.pandas ./exp-octave/irm1 ./exp-octave/irm2 ./exp-octave/ibm2 ./exp-octave/ibm1 ./exp-octave/*-oct-20-*
$pybin oracle_eval/aggregate.py --out=exp-octave/data_275.pandas ./exp-octave/irm1 ./exp-octave/irm2 ./exp-octave/ibm2 ./exp-octave/ibm1 ./exp-octave/*-oct-275-*
$pybin oracle_eval/aggregate.py --out=exp-octave/data_327.pandas ./exp-octave/irm1 ./exp-octave/irm2 ./exp-octave/ibm2 ./exp-octave/ibm1 ./exp-octave/*-oct-327-*
$pybin oracle_eval/aggregate.py --out=exp-octave/data_57.pandas ./exp-octave/irm1 ./exp-octave/irm2 ./exp-octave/ibm2 ./exp-octave/ibm1 ./exp-octave/*-oct-57-*
$pybin oracle_eval/aggregate.py --out=exp-octave/data_80.pandas ./exp-octave/irm1 ./exp-octave/irm2 ./exp-octave/ibm2 ./exp-octave/ibm1 ./exp-octave/*-oct-80-*

$pybin oracle_eval/boxplot.py ./exp-octave/data_20.pandas ./exp-octave/boxplot_20.pdf
$pybin oracle_eval/boxplot.py ./exp-octave/data_275.pandas ./exp-octave/boxplot_275.pdf
$pybin oracle_eval/boxplot.py ./exp-octave/data_327.pandas ./exp-octave/boxplot_327.pdf
$pybin oracle_eval/boxplot.py ./exp-octave/data_57.pandas ./exp-octave/boxplot_57.pdf
$pybin oracle_eval/boxplot.py ./exp-octave/data_80.pandas ./exp-octave/boxplot_80.pdf

#$pybin oracle_eval/aggregate.py ./exp-mel/* --out=exp-mel/data.pandas
#$pybin oracle_eval/boxplot.py ./exp-mel/data.pandas ./exp-mel/boxplot.pdf
#
#$pybin oracle_eval/aggregate.py ./exp-bark/* --out=exp-bark/data.pandas
#$pybin oracle_eval/boxplot.py ./exp-bark/data.pandas ./exp-bark/boxplot.pdf
#
#$pybin oracle_eval/aggregate.py ./exp-log/* --out=exp-log/data.pandas
#$pybin oracle_eval/boxplot.py ./exp-log/data.pandas ./exp-log/boxplot.pdf
