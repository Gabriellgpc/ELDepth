# FROM pytorchlightning/pytorch_lightning:base-cuda-py3.9-torch1.10
FROM tensorflow/tensorflow:2.10.0-gpu

# A workaround in case any problem with cuda
# RUN rm /etc/apt/sources.list.d/cuda.list
# RUN apt-key del 7fa2af80
# RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/7fa2af80.pub

LABEL maintainer="Condados"
LABEL version="v2.0.0"
LABEL description="My basic docker image"

# Install base packages
RUN apt-get update --fix-missing && apt-get install -y \
    v4l-utils\
    # to avoid cv2 import error
    ffmpeg libsm6 libxext6     \
    wget bzip2 ca-certificates \
    cmake build-essential      \
    byobu                      \
    curl                       \
    git-core                   \
    htop                       \
    pkg-config                 \
    python3-dev                \
    python3-pip                 \
    python3-setuptools          \
    python3-virtualenv          \
    unzip                      \
    graphviz                   \
    iputils-ping               \
    && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

ENV HOME=/home
RUN chmod -R a+rwx /home

# Here we get all python packages.
# There's substantial overlap between scipy and numpy that we eliminate by
# linking them together. Likewise, pip leaves the install caches populated which uses
# a significant amount of space. These optimizations save a fair amount of space in the
# image, which reduces start up time.

COPY ./requirements.txt /workspace/requirements.txt

# Install pip packages
RUN python3 -m pip install --upgrade pip
RUN pip3 install --no-cache-dir --upgrade -r /workspace/requirements.txt

# Set some environment variables. PYTHONUNBUFFERED keeps Python from buffering our standard
# output stream, which means that logs can be delivered to the user quickly. PYTHONDONTWRITEBYTECODE
# keeps Python from writing the .pyc files which are unnecessary in this case. We also update
# PATH so that the serve program are found when the container is invoked.

ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE
ENV PATH="/workspace:${PATH}"
# Configure python path for custom modules
ENV PYTHONPATH "${PYTHONPATH}:/workspace"

# Configure default shell
ENV SHELL=/bin/bash