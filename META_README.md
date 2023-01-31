# xumx-sliCQ-V2

## Current running command

```
sevagh:xumx-sliCQ-V2 $ docker run -it --gpus=all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -v /home/sevagh/Music/MUSDB18-HQ/:/MUSDB18-HQ -v /home/sevagh/repos/xumx-sliCQ-V2:/xumx-sliCQ-V2 -p 6006:6006 xumx-slicq-v2 python -m xumx_slicq_v2.training --debug  --valid-chunk-dur=60 --batch-size=8
```

## Goals

1. Improve SDR of xumx-sliCQ
1. Adapt for HAAQI: http://cadenzachallenge.org/docs/cadenza1/Software/cc1_baseline
    1. with cuSignal

## Guides

* <https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html>
* <https://github.com/google-research/tuning_playbook>
* <https://karpathy.github.io/2019/04/25/recipe/>

## Code roadmap

1. Full iNSGT gradient training (~200 hours, >1 week)
1. Apply HAAQI metrics calculations for time-domain
    * implement [HAAQI](https://github.com/claritychallenge/clarity/blob/main/clarity/evaluator/haaqi/haaqi.py)
        * with cuSignal
    * replace SISDR with HAAQI with a flag
