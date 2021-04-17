#!/usr/bin/env bash

MUSDB_PATH=/home/sevagh/TRAINING-MUSIC/MUSDB18-HQ python IRM_CQT.py --alpha=1 --eval_dir ./IRM1-CQT/
MUSDB_PATH=/home/sevagh/TRAINING-MUSIC/MUSDB18-HQ python IRM_CQT.py --alpha=2 --eval_dir ./IRM2-CQT/

MUSDB_PATH=/home/sevagh/TRAINING-MUSIC/MUSDB18-HQ python IRM.py --alpha=1 --eval_dir ./IRM1/
MUSDB_PATH=/home/sevagh/TRAINING-MUSIC/MUSDB18-HQ python IRM.py --alpha=2 --eval_dir ./IRM2/
