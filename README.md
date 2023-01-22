# xumx-sliCQ-V2

intro blah blah

## Prerequisites

* CUDA
* Docker
* Nvidia-docker runtime

Caveats: things will be `root`y

## Getting started

1. Build the Docker container (containing the CUDA toolkit, PyTorch, other dependencies, etc.)

```
$ docker build -t "xumx-slicq-v2" .
```

2. Run training on dataset `/MUSDB18-HQ`, save model in `/model`, visit tensorboard at <http://127.0.0.1:6006/>

```
$ docker run --rm -it \
    --gpus=all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
    -v /path/to/MUSDB18-HQ/dataset:/MUSDB18-HQ
    -v /path/to/save/trained/model:/model
    -p 6006:6006
    xumx-slicq-v2 \
    python -m xumx_slicq_v2.training --help
```

**N.B.** if your model path already contains a trained model, training will continue from the saved checkpoint

3. Run inference on `/input` tracks with `/model`, save into `/output`

```
$ docker run --rm -it \
    --gpus=all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
    xumx-slicq-v2 \
    -v /path/to/input/tracks:/input
    -v /path/to/output/tracks:/output
    -v /path/to/trained/model:/model
    python -m xumx_slicq_v2.inference --help
```
