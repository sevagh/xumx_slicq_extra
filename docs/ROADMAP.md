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

*training repo*
* xumx-sliCQ-V2-training: this repo
    * keep training code, blendmodels, single Dockerfile, etc.
    * add some plot-spectrograms option with `_overlap-add-slicq` (private function), for visualizations
    * add latex files etc. for future papers
    * HAAQI stuff happens here
1. Add MSE+HAAQI (new pretrained_model/mse-haaqi) with new mcoef
    1. train new variant
    1. use cuSignal for HAAQI
        * http://cadenzachallenge.org/docs/cadenza1/Software/cc1_baseline
        * https://github.com/claritychallenge/clarity/blob/main/clarity/evaluator/haaqi/haaqi.py
* Cadenza challenge registration issues
* README to describe all the cool things (and not so cool things)
    nvcr, training, blending, <https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html>

*public repo*
* xumx-sliCQ-V2 (for public) with cut down code, 1 pretrained MSE model, xumx-config=1
    1. Inference = '__main__.py'; ensure it works; CPU or GPU inference with outputting files (for demos etc.) is fine
    1. Start working on README, paper materials
    1. create slim Dockerfile for pytorch runtime inference
