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
1. Cross-target skip connection: *TBD*; not looking good; ditch skip connections because you don't understand them!
<https://file.techscience.com/ueditor/files/csse/TSP_CSSE-44-3/TSP_CSSE_29732/TSP_CSSE_29732.pdf>
<https://arxiv.org/pdf/1606.08921.pdf>
1. Mixing frequency bins: global bottleneck layer (with or without skip conn)
    ```
    # list
    encoded, skip_conn = sliced_umx.encoder()

    # global bottleneck
    encoded_concat = concat or whatever
    # try this: https://stats.stackexchange.com/a/552170
    encoded_concat = self.bottleneck(encoded_concat)
    encoded = deconcat
        
    decoded, masks = sliced_umx.decoder(encoded, skip_conn)
    ```
1. New slicqt with 10.17 wiener oracle (vs. 10.14): 'bark', 288, 43.39999999999988
1. try smaller wiener window than 5000 for effect on memory/swap
1. Lighter model weights: bandwidth + dummy time bucket
1. Option to train/infer without wiener (but don't actually use it)

## Training/internal details housekeeping

1. visualization.py: spectrogram plotting code (overlap-add, flatten, per-block)
    * figure out MLS bug etc.
1. Training README
    * how its a "toolbox" for slicqt-based demixing (phasemix, wiener)
    * dockerized setup + example commands for everything, w/ cpu/gpu
    * nvcr + perf tuning: <https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html>
    * optuna + optuna dashboard/screenshots
    * tensorboard + training UI/screenshots

## Inference/external publication housekeeping

1. TensorRT export script
1. Inference.py (and by association `__main__.py` work); CPU, GPU, TensorRT
    1. realtime inference script? measure time/latency w/ causal inputs
1. tag as "v1.0.0a"
1. README (no training details), seed of future papers
1. Dockerfile.slim for xumx-slicq-v2-slim pytorch/runtime inference container on DockerHub
    * publish to dockerhub with github actions
