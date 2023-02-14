# xumx-sliCQ-V2

## Current running command

```
docker run --rm -it \
    --privileged --gpus=all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
    -v /home/sevagh/thirdparty-repos/MUSDB18-HQ:/MUSDB18-HQ \
    -v /home/sevagh/thirdparty-repos/xumx-sliCQ-V2:/xumx-sliCQ-V2 \
    -v /home/sevagh/thirdparty-repos/xumx-sliCQ-V2/trained-model:/model \
    -p 6006:6006 \
    xumx-slicq-v2 \
    python -m xumx_slicq_v2.training --batch-size=256 --batch-size-valid=3
```

## Goals

start cleanups, deal with Git-LFS warnings
    - dont need janky failure pretrained models

looks like nb-workers, lower is better?

* ensemble xumx-sliCQ model! create methods `def vocals/bass/drums/other` and mix and match them
    * combine weights from different models! and mix and match wiener methods
    * first test both in conjunction, _then_ create a frankenloader to save blended weights

blended: bass from mse-sdr, vocals/drums/other from mse
try config 2 anyway
then, blend + 0/1/2
then, blend + save

blending was an accident; mixed up median scores with Zeno - Signs
    - try again

evaluation: better tqdm/less print statements now that i know things are working

xumx-sliCQ-V2-private
then xumx-sliCQ-V2-training
then xumx-sliCQ-V2 clean inference
    create clean inference repo with divide described below

1. Ownership, license, etc.
    1. xumx-sliCQ-V2-private: private, later public archive (and add README and point to from xumx-sliCQ-V2)
    1. xumx-sliCQ-V2: public release repo, don't publish this one, for clean commit
    1. create xumx_slicq_v2_extra module for code i won't be using (but don't want to delete?)
        1. xumx_slicq_v2_extra (or devel) (w/ matplotlib)
            1. put back other frequency scales??
            1. remove unused configs e.g. slicqt-wiener? keep highest performing models, prune other code
            1. add some plot-spectrograms option with `_overlap-add-slicq` (private function), for visualizations
            1. evaluation goes in extra?? benchmark?? training??
                1. evaluation, loss, training, data, nsgt_extra (frequency scales + plotting + slicqt-wiener)
                1. HAAQI + cuSignal goes here too, + auraloss + sevagh/sigsep-mus-eval + scikit-learn + tensorboard + torchinfo + gitpython + musdb
                    1. get rid of gitpython
        1. create dockerfile-slim for pytorch runtime inference
            1. with only xumx_slicq_v2
            1. only cp best pretrained model into it (3x70 = 210MB total storage)

* best MSE loss to beat: "best_loss": 0.07773791626095772" (epoch 334)
    * SDR with STFT-wiener-EM = 4.1 dB ('mse' variant, config 1)
* next MSE-SDR model: 
    * SDR with STFT-wiener-EM = 

1. Current: NN
    1. Train next variant (MSE-SDSDR with --sdsdr-mcoef=0.01)
    1. store: pretrained_model/{mse, mse-sdsdr}
    1. test all of them:
        mse: config 0, 1, 2
        mse-sdr: config 0, 1, 2
        move worse code (slicqt-wiener-EM) to xumx_slicq_v2_extra
    1. Ensure `inference.py` works; CPU or GPU inference with outputting files (for demos etc.) is fine

1. Next: competition; MSE+HAAQI (new mcoef)
    1. Start working on README, paper materials
    1. Cadenza challenge registration issues
    1. Add MSE+HAAQI (new pretrained_model/mse-haaqi)
        1. train new variant
        1. use cuSignal for HAAQI
            * http://cadenzachallenge.org/docs/cadenza1/Software/cc1_baseline
            * https://github.com/claritychallenge/clarity/blob/main/clarity/evaluator/haaqi/haaqi.py
        1. submit all 3

## Guides

* <https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html>
