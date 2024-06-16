# Description: Dockerfile for JAX with CUDA support
FROM 12.2.2-cudnn8-devel-ubuntu20.04

# Set the working directory
WORKDIR /workspace

# Set the environment variables
ENV DEBIAN_FRONTEND=noninteractive

# Install the required packages and python
RUN apt-get update && \
    apt-get install -y curl software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa

# actual installation of python
RUN apt-get update && \
    apt-get install -y python3.11 python3.11-distutils libglfw3-dev git

# Install pip
RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
    python3.11 get-pip.py --force-reinstall && \ 
    rm -rf /var/lib/apt/lists/*

# Copy the requirements file
COPY requirements.txt .

# Install the required packages
RUN python3.11 -m pip install -r requirements.txt

# Install JAX with CUDA support.
RUN python3.11 -m pip install --upgrade \
    pip install -U "jax[cuda12]" \
    optax

# Set the environment variables
ENV PYGLFW_PREVIEW=1

# Set the default python version
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1
