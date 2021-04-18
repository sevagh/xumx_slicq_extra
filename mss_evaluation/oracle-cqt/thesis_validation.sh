#!/usr/bin/env bash

rm sisecc.pandas
rm -rf ./experiment-02

export MUSDB_MAX_TRACKS=2
export MUSDB_PATH=/home/sevagh/TRAINING-MUSIC/MUSDB18-HQ

python evaloracle/IRM.py --alpha=1 --eval_dir ./experiment-02/IRM1/
#python evaloracle/IRM.py --alpha=2 --eval_dir ./experiment-02/IRM2/
python evaloracle/IRM.py --binary-mask --alpha=1 --eval_dir ./experiment-02/IBM1/
#python evaloracle/IRM.py --binary-mask --alpha=2 --eval_dir ./experiment-02/IBM2/

python evaloracle/IRM.py --use-cqt --alpha=1 --eval_dir ./experiment-02/IRM1-CQT1/
#python evaloracle/IRM.py --use-cqt --alpha=2 --eval_dir ./experiment-02/IRM2-CQT1/
python evaloracle/IRM.py --use-cqt --alpha=1 --eval_dir ./experiment-02/IBM1-CQT1/
#python evaloracle/IRM.py --use-cqt --alpha=2 --eval_dir ./experiment-02/IBM2-CQT1/

python evaloracle/aggregate.py experiment-02/* --out=sisecc.pandas
python evaloracle/boxplot.py ./sisecc.pandas experiment-02/oracle_boxplot.pdf
