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

## NN architecture/training

1. Starting point: 28MB, 4.24 dB (full bw, block wiener, complex MSE loss = 0.0395)
1. Optuna hyperparams (50,51,4): 60MB, 4.35 dB (0.0390)
1. Mask sum loss: 0.0405, 4.4 dB
1. Mixing frequency bins: global bottleneck layer (still 60MB!)
    1. 0.0387 loss, 4.37475 dB (lower SDR performance??)
1. next training **in-progress!** 
    1. add back training SD-SDR loss (without big gradient)
    1. new wiener oracle: 10.17 (vs. 10.14): 'bark', 288, 43.39999999999988
    1. Use leaky ReLU

## Post-trained model code/tasks

1. Model pruning to save space? iterative + finetuning (like Optuna)
    1. SDR-based pruning? <https://github.com/EIDOSlab/simplify>?
        <https://github.com/spellml/resnext50-panda/blob/master/notebooks/pruning.ipynb>
1. TensorRT export script
1. Inference.py (and by association `__main__.py` work); CPU, GPU, TensorRT
    1. Lighter computation: try smaller wiener window than 5000 for effect on memory/swap?
    1. realtime inference script? measure time/latency w/ causal inputs

## Housekeeping

1. visualization.py: spectrogram plotting code (overlap-add, flatten, per-block)
    * figure out MLS bug etc.
1. Training README
    * Mermaid diagrams?
    * how its a "toolbox" for slicqt-based demixing (phasemix, wiener)
    * dockerized setup + example commands for everything, w/ cpu/gpu
    * nvcr + perf tuning: <https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html>
    * optuna + optuna dashboard/screenshots
    * tensorboard + training UI/screenshots
1. README (no training details), seed of future papers
1. Dockerfile.slim for xumx-slicq-v2-slim pytorch/runtime inference container on DockerHub
    * publish to dockerhub with github actions
1. code_quality.sh: implement suggestions, cleanups, etc.
1. tag as "v1.0.0a"
