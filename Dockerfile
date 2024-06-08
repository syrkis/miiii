FROM nvidia/cuda:11.1.1-cudnn8-devel-ubuntu20.04

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

COPY requirements.txt .

RUN python3.11 -m pip install -r requirements.txt

RUN python3.11 -m pip uninstall optax jaxlib jax

RUN python3.11 -m pip install "jax[cuda11_pip]" jaxlib==0.1.70+cuda111 -f https://storage.googleapis.com/jax-releases/jax_releases.html optax

RUN python3.11 -m pip install tensorflow_datasets opencv-python pycocotools

ENV PYGLFW_PREVIEW=1

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1