# xumx-sliCQ-V2

## Target to beat

Time for epoch: ~2:20

## Goals

1. Adapt xumx-sliCQ for HAAQI: http://cadenzachallenge.org/docs/cadenza1/Software/cc1_baseline
2. Improve SDR of xumx-sliCQ
3. Showcase NVIDIA/RAPIDS tools/SDKs for kgmon

### Showcase NVIDIA tools

1. NGC pytorch for enhanced GPU acceleration
    1. Data: DALI ETL (load MUSDB18-HQ faster)
    2. Training: cuDNN (nothing to do, pytorch as usual)
    3. Inference: TensorRT
2. cuFFT
3. cuSignal

## ML/DL Training decisions

Using guides:
* <https://github.com/google-research/tuning_playbook>
* <https://karpathy.github.io/2019/04/25/recipe/>

## Realistic roadmap

2. Prepare optimal training/tuning environment
    * Tensorboard to show spectrograms, losses, etc.
    * Prepare a few for-overfitting sets of data
3. Apply HAAQI metrics calculations for time-domain
    * implement [HAAQI](https://github.com/claritychallenge/clarity/blob/main/clarity/evaluator/haaqi/haaqi.py)
        * with cuSignal and/or cuFFT?
4. Address overlap issue: normal cqlog sliCQT, delete deoverlap (focus on mega-slices)
5. Address ragged tensor issue (interpolated matrix form - separate magnitude/phase interpolation)
    * improve performance with cuFFT/cuSignal
    * code files: `xumx_slicq_v2/nsgt/`
    * address invertible interpolation/matrixform for CQNSGT (rasterization?) from paper
6. Use [Optuna](https://optuna.org/) to tune sliCQT hyperparameters
7. Apply NVIDIA NGC container tools/tooling (DALI data loading, TensorRT inference, cuFFT, cuSignal)

## Code roadmap

Running/packaging:
- nvidia-docker2 runtime for reproducible training
