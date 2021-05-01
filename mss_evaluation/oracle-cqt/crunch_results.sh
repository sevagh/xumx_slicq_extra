#!/usr/bin/env bash

pybin="/home/sevagh/venvs/thesis/bin/python3"

$pybin oracle_eval/aggregate.py --out=exp-irm-mel/data.pandas ./exp-irm-mel/*
$pybin oracle_eval/aggregate.py --out=exp-irm-mel/data_small.pandas ./exp-irm-mel/irm1-2048 ./exp-irm-mel/irm2-2048 ./exp-irm-mel/ibm2-2048 ./exp-irm-mel/ibm1-2048 ./exp-irm-mel/*-mel-234-275*

$pybin oracle_eval/boxplot.py --single  ./exp-irm-mel/data.pandas ./exp-irm-mel/boxplot.pdf
$pybin oracle_eval/boxplot.py ./exp-irm-mel/data_small.pandas ./exp-irm-mel/boxplot_small.pdf
