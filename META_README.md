# xumx-sliCQ-V2

## Current running command

```
sevagh:xumx-sliCQ-V2 $ docker run -it --gpus=all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -v /home/sevagh/Music/MUSDB18-HQ/:/MUSDB18-HQ -v /home/sevagh/repos/xumx-sliCQ-V2:/xumx-sliCQ-V2 -p 6006:6006 xumx-slicq-v2 python -m xumx_slicq_v2.training --debug  --valid-chunk-dur=60 --batch-size=8
```

## Goals

1. Improve SDR of xumx-sliCQ
1. Adapt for HAAQI: http://cadenzachallenge.org/docs/cadenza1/Software/cc1_baseline
1. Showcase NVIDIA/RAPIDS tools/SDKs for kgmon
    1. NGC pytorch container with cuSignal installed
    1. Improve performance of training/inference with DALI, TensorRT, cuSignal, cuFFT, etc.

## ML/DL Training decisions

Using guides:
* <https://github.com/google-research/tuning_playbook>
* <https://karpathy.github.io/2019/04/25/recipe/>

## Realistic roadmap

1. Train fully with iNSGT gradients
1. Remove bandwidth parameter, train again
1. Apply HAAQI metrics calculations for time-domain
    * implement [HAAQI](https://github.com/claritychallenge/clarity/blob/main/clarity/evaluator/haaqi/haaqi.py)
    * replace SISDR with HAAQI (with a flag?) or mix both
1. Apply NVIDIA NGC container tools/tooling (DALI data loading, TensorRT inference, cuFFT, cuSignal)
1. Look into where cuFFT or cuSignal can speed things up
