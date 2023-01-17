## Roadmap

http://cadenzachallenge.org/docs/cadenza1/Software/cc1_baseline
    * cuDemixUtils - with cuSignal
        * sdr, isr, sar, sir, haaqi, mse (for both loss + eval)
        * HAAQI: https://github.com/claritychallenge/clarity/blob/main/clarity/evaluator/haaqi/haaqi.py
    * cuNSGT - with cuSignal again (or cuFFT)
        - fresh take on original NSGT code, written for performance
        - address invertible interpolation/matrixform for CQNSGT (rasterization?)
        - address deoverlap

## Goals

1. Adapt xumx-sliCQ for HAAQI
2. Showcase RAPIDS tools or other NVIDIA SDKs (cuSignal, cuFFT) for kgmon
3. Improve SDR of xumx-sliCQ
4. Have adaptible HAAQI or SDR variants of xumx-sliCQ
5. Use mix of SAR/SIR/SDR/ISR for training

## Initial plan

Create repo structure:
* xumx_sliCQ_V2
    - the deep learning model
    - original repo: sevagh/xumx-sliCQ
* cuNSGT
    - cuSignal/cuFFT fresh take on original NSGT code
    - address invertible interpolation/matrixform for CQNSGT (rasterization?)
        - look into Essentia CQT for rasterization
    - address deoverlap (giant slice!)
    - less focus on frequency scales, stick with CQ with giant slice
    - original repo: sevag/nsgt
* cuDemixUtils
    - cuSignal/cuFFT/CuPY all accelerated
    - sdr, isr, sar, sir, haaqi, mse (losses, evaluation)
    - original repo: sevagh/sigsep-mus-eval
* extra from repo: sevagh/xumx_slicq_extra

meta-tools:
- Python: pyproject.toml, pytest
- mamba env + pyproject.toml for each subproject
- poetry on main binary xumx-sliCQ-V2
- nvidia-docker2 runtime for reproducible
