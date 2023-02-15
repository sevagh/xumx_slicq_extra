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

* xumx-sliCQ-V2-scrapyard: current (can always make it public again)
* then xumx-sliCQ-V2 (public) with inference/extra split
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
* MSE-SDR, blending: insignificant gains (4.16 vs. 4.1 dB? not worth the complexity)

1. Next NN design: concatenate _some_ tf blocks with zero-padding where appropriate
1. Next NN design: bias=False+BatchNorm+ReLU (for symmetry) vs. Sigmoid
1. evaluation: better tqdm/less print statements now that i know things are working
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
