# xumx-sliCQ-V2

intro blah blah

## Prerequisites

* CUDA
* Docker
* Nvidia-docker runtime

## Getting started

1. Build the Docker container (containing the CUDA toolkit, PyTorch, other dependencies, etc.)

```
docker build -t "xumx-slicq-v2" .
```

2. Run training

```
docker run -it --gpus=all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 xumx-slicq-v2 python -m xumx_slicq_v2.training --help
```

3. Run inference (i.e. separate music into stems)

```
docker run -it --gpus=all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 xumx-slicq-v2 python -m xumx_slicq_v2.inference --help
```
