FROM nvidia/cuda:12.0.0-cudnn8-devel-ubuntu20.04

WORKDIR /workspace

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y curl software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa

RUN apt-get update && \
    apt-get install -y python3.11 python3.11-distutils libglfw3-dev git

RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
    python3.11 get-pip.py --force-reinstall && \ 
    rm -rf /var/lib/apt/lists/*

RUN python3.11 -m pip install --upgrade \
    miiiii "jax[cuda12]" optax