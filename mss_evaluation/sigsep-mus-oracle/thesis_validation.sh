#!/usr/bin/env bash

rm sisecc.pandas
rm -rf ./experiment-02

MUSDB_PATH=/home/sevagh/TRAINING-MUSIC/MUSDB18-HQ python IRM.py --alpha=1 --eval_dir ./experiment-02/IRM1/
MUSDB_PATH=/home/sevagh/TRAINING-MUSIC/MUSDB18-HQ python IRM.py --alpha=2 --eval_dir ./experiment-02/IRM2/
MUSDB_PATH=/home/sevagh/TRAINING-MUSIC/MUSDB18-HQ python IBM.py --alpha=2 --eval_dir ./experiment-02/IBM2/
MUSDB_PATH=/home/sevagh/TRAINING-MUSIC/MUSDB18-HQ python IBM.py --alpha=1 --eval_dir ./experiment-02/IBM2/

MUSDB_PATH=/home/sevagh/TRAINING-MUSIC/MUSDB18-HQ python IRM_CQT.py --alpha=1 --eval_dir ./experiment-02/IRM1-CQT/
MUSDB_PATH=/home/sevagh/TRAINING-MUSIC/MUSDB18-HQ python IRM_CQT.py --alpha=2 --eval_dir ./experiment-02/IRM2-CQT/
MUSDB_PATH=/home/sevagh/TRAINING-MUSIC/MUSDB18-HQ python IBM_CQT.py --alpha=1 --eval_dir ./experiment-02/IBM1-CQT/
MUSDB_PATH=/home/sevagh/TRAINING-MUSIC/MUSDB18-HQ python IBM_CQT.py --alpha=2 --eval_dir ./experiment-02/IBM2-CQT/

python aggregate.py experiment-02/* --out=sisecc.pandas
python boxplot.py ./sisecc.pandas
