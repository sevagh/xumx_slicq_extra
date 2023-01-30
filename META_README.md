# xumx-sliCQ-V2

## Current running command

```
sevagh:xumx-sliCQ-V2 $ docker run -it --gpus=all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -v /home/sevagh/Music/MUSDB18-HQ/:/MUSDB18-HQ -v /home/sevagh/repos/xumx-sliCQ-V2:/xumx-sliCQ-V2 -p 6006:6006 xumx-slicq-v2 python -m xumx_slicq_v2.training --debug  --valid-chunk-dur=60 --batch-size=8
```

## Goals

1. Improve SDR of xumx-sliCQ
1. Adapt for HAAQI: http://cadenzachallenge.org/docs/cadenza1/Software/cc1_baseline
1. Showcase NVIDIA/RAPIDS tools/SDKs for kgmon: NGC pytorch (+ DALI + TensorRT)

## Guides

* <https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html>
* <https://github.com/google-research/tuning_playbook>
* <https://karpathy.github.io/2019/04/25/recipe/>

## Code roadmap

1. Improve: Danna-Sep (differentiable MWF before SISDR loss + complex MSE loss)
    1. Improve: multi-sliCQT
    1. MWF/Norbert after waveforms
    1. Complex MSE loss
1. Apply HAAQI metrics calculations for time-domain
    * implement [HAAQI](https://github.com/claritychallenge/clarity/blob/main/clarity/evaluator/haaqi/haaqi.py)
        * with cuSignal
    * replace SISDR with HAAQI with a flag
1. Apply NVIDIA NGC container tools/tooling (DALI + TensorRT) wherever it can speed things up

## Dlprof guide

Run:
```
# dlprof --mode=pytorch --output_path /dlprof_out  --reports=summary --formats=json python -m xumx_slicq_v2.training --debug --dlprof --samples-per-track=16
```

View:
```
# dlprofviewer /dlprof_out/dlprof_dldb.sqlite -b 0.0.0.0
```
