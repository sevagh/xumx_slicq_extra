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

* Cadenza challenge registration issues

## Goals


*effort 1: training repo*
* xumx-sliCQ-V2-training: this repo
    * keep training code, blendmodels, single Dockerfile, etc.
    * add some plot-spectrograms option with `_overlap-add-slicq` (private function), for visualizations
    * add latex files etc. for future papers

* 28MB v1 model would be killer
    * New training
        * Differentiable sliCQT-Wiener w/ complex-MSE, squeeze more juice from network, v1 28MB
        * looking meh but we have a complex-valued loss baseline: 0.0395
        * now mag-phase umx; Tanh activation for phase??
        * double the channels; 50, 110 :shrug:? or 32/64 if you must
        * details: add back ComplexSDR (for both real and complex)
            * dont forget loss error revelation: unmix->mse-loss->waveform->sdr loss = mse loss missing gradients
        * leakyrelu??
    * clean up unused shit
* TensorRT save script (ala blendmodels)
* README to describe all the cool things (and not so cool things)
    nvcr, training, blending, <https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html>

*effort 2: inference/public repo*
    * use warp + potential C++ kernels (C++ for C++ sake is a bad idea, remember) for packed-nsgt
        * must be compatible with regular nsgt!
    * TensorRT script; load + use packed-nsgt with realtime inputs (and offline, same script)
    * provide measurements/SDR docs for it all

    *public repo*
    * xumx-sliCQ-V2 (for public) with cut down code: single best model + Wiener
        1. Inference = '__main__.py'; ensure it works; CPU or GPU inference with outputting files (for demos etc.) is fine
        1. Start working on README, paper materials
        1. create slim Dockerfile for pytorch runtime inference
