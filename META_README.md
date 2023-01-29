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

1. Train fully with iNSGT gradients
1. Optimize more
    1. `code_quality.sh`: delete unused code with `vulture`, format with `black`, `perflint`, `pylint`
    1. dlprof/nvtx, tf32, bfloat16/amp, no_grad, etc.
1. If good at this point, delete dlprof stuff
1. Apply HAAQI metrics calculations for time-domain
    * implement [HAAQI](https://github.com/claritychallenge/clarity/blob/main/clarity/evaluator/haaqi/haaqi.py)
        * with cuSignal
    * replace SISDR with HAAQI (with a flag?) or mix both
1. Apply NVIDIA NGC container tools/tooling (DALI + TensorRT) wherever it can speed things up
1. Apply different sliCQ per-target for even more gains
    1. use string representation of slicq parameters
        `--fscales <dbov>` etc.
    1. Optuna?

## Dlprof guide

Run:
```
# dlprof --mode=pytorch --output_path /dlprof_out  --reports=summary --formats=json python -m xumx_slicq_v2.training --debug --dlprof --samples-per-track=16
```

View:
```
# dlprofviewer /dlprof_out/dlprof_dldb.sqlite -b 0.0.0.0
```
