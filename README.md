# xumx-sliCQ-V2

## Goals

1. Adapt xumx-sliCQ for HAAQI: http://cadenzachallenge.org/docs/cadenza1/Software/cc1_baseline
2. Showcase RAPIDS tools or other NVIDIA SDKs (cuSignal, cuFFT) for kgmon
3. Improve SDR of xumx-sliCQ

## Code roadmap

`xumx_slicq_v2/{metrics.py,loss.py}`:
* implement [HAAQI](https://github.com/claritychallenge/clarity/blob/main/clarity/evaluator/haaqi/haaqi.py)
* implement mixed SDR/SAR/SIR/ISR loss
* MSE loss
* mega-ultimate-everything-loss??
* cuSignal/cuFFT

`xumx_slicq_v2/nsgt/`:
* fresh take on original NSGT code, written for performance
* address invertible interpolation/matrixform for CQNSGT (rasterization?)
* address deoverlap (_or_ use giant slices)
* cuSignal/cuFFT

Running/packaging:
- nvidia-docker2 runtime for reproducible training
- wheel stuff (setup/pyproject.toml/poetry)

`xumx_slicq_v2/model.py`:
* use a single NSGT
* forget weird param search, use a basic cqlog scale
