#!/usr/bin/env sh

#python scripts/train.py --root /home/sevagh/TRAINING-MUSIC/MUSDB18-HQ/ --target vocals --is-wav --nb-workers=4 --batch-size=12 --epochs=1 --output umx-nsgt --debug
python -m kernprof -l -v  scripts/train.py --root /home/sevagh/TRAINING-MUSIC/MUSDB18-HQ/ --target vocals --is-wav --nb-workers=4 --batch-size=12 --epochs=1 --output umx-nsgt --debug

#python scripts/train.py --root /home/sevagh/TRAINING-MUSIC/MUSDB18-HQ/ --target drums --is-wav --nb-workers=4 --batch-size=32 --epochs=1 --output umx-nsgt --debug --no-cuda
#python scripts/train.py --root /home/sevagh/TRAINING-MUSIC/MUSDB18-HQ/ --target bass --is-wav --nb-workers=4 --batch-size=32 --epochs=1 --output umx-nsgt --debug --no-cuda
#python scripts/train.py --root /home/sevagh/TRAINING-MUSIC/MUSDB18-HQ/ --target other --is-wav --nb-workers=4 --batch-size=32 --epochs=1 --output umx-nsgt --debug --no-cuda
#
#python -m openunmix.evaluate --outdir=./ests-umx-nsgt --evaldir=./results-umx-nsgt --model=./umx-nsgt/ --cores 1 --no-cuda
