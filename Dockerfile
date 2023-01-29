ARG XUMX_SLICQ_V2_VERSION="0.1.0"

###############
# DEVEL STAGE #
###############
FROM nvcr.io/nvidia/pytorch:22.12-py3 as devel
ARG XUMX_SLICQ_V2_VERSION
ENV XUMX_SLICQ_V2_VERSION="${XUMX_SLICQ_V2_VERSION}"

RUN python -m pip install --upgrade pip

# build a wheel for torchaudio from source to correspond to $PYTORCH_VERSION from nvcr
RUN git clone https://github.com/pytorch/audio /torchaudio
WORKDIR /torchaudio
RUN python -m pip wheel --no-build-isolation --no-deps ./ --wheel-dir /wheelhouse

#################
# RUNTIME STAGE #
#################
FROM nvcr.io/nvidia/pytorch:22.12-py3 as runtime
ARG XUMX_SLICQ_V2_VERSION
ENV XUMX_SLICQ_V2_VERSION="${XUMX_SLICQ_V2_VERSION}"

RUN export DEBIAN_FRONTEND="noninteractive" && apt-get update -y && \
	apt-get install -y ffmpeg libavcodec-dev libavdevice-dev libavfilter-dev libavformat-dev libavresample-dev libavutil-dev

RUN python -m pip install --upgrade pip

COPY --from=devel /wheelhouse /wheelhouse

RUN git clone https://github.com/fakufaku/fast_bss_eval /fast_bss_eval &&\
	cd /fast_bss_eval && python -m pip install -e .

# install xumx-slicq-v2 from source to get its dependencies
COPY . /xumx-sliCQ-V2
WORKDIR /xumx-sliCQ-V2
RUN python -m pip install --pre -e . --find-links /wheelhouse
