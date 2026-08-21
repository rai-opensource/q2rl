# syntax=docker/dockerfile:1.7-labs
ARG CUDA_VER=12.4.1
ARG UBUNTU_VER=22.04
FROM nvidia/cuda:${CUDA_VER}-cudnn-devel-ubuntu${UBUNTU_VER}

USER root
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ARG PYTHON_VER=3.10
ARG UV_VERSION=0.11.8
ARG DEBIAN_FRONTEND="noninteractive"
ENV LANG="C.UTF-8"
ENV LC_ALL="C.UTF-8"

RUN apt-get -o Acquire::Retries=5 update && apt-get install -y --no-install-recommends \
    bash \
    build-essential \
    ca-certificates \
    cmake \
    ffmpeg \
    git \
    git-lfs \
    libavcodec-dev \
    libavdevice-dev \
    libavfilter-dev \
    libavformat-dev \
    libavutil-dev \
    libegl1-mesa-dev \
    libffi-dev \
    libgl1-mesa-dri \
    libgl1-mesa-glx \
    libglew-dev \
    libglfw3 \
    libglfw3-dev \
    libglu1-mesa \
    libosmesa6-dev \
    libssl-dev \
    libswscale-dev \
    patchelf \
    python${PYTHON_VER} \
    python${PYTHON_VER}-dev \
    python-is-python3 \
    python3-pip \
    python3.10-venv \
    tzdata \
    unzip \
    wget \
    x11-utils \
    xvfb \
    xz-utils \
    zip \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

ADD https://astral.sh/uv/${UV_VERSION}/install.sh /uv-installer.sh
RUN sh /uv-installer.sh && rm /uv-installer.sh
ENV PATH="/root/.local/bin/:$PATH"
ENV UV_LINK_MODE=copy
ENV UV_RELOCATABLE=1
ENV UV_PROJECT_ENVIRONMENT=/opt/venv

ENV CUDA_HOME=/usr/local/cuda
ENV PATH=/usr/local/cuda/bin:$PATH

# MuJoCo 2.1.0 for mujoco-py / adroit (mj_envs)
WORKDIR /root/.mujoco
RUN wget -q https://github.com/deepmind/mujoco/releases/download/2.1.0/mujoco210-linux-x86_64.tar.gz \
    && tar -xzf mujoco210-linux-x86_64.tar.gz \
    && rm mujoco210-linux-x86_64.tar.gz
WORKDIR /

ENV MUJOCO_PY_MUJOCO_PATH=/root/.mujoco/mujoco210
ENV LD_LIBRARY_PATH="/root/.mujoco/mujoco210/bin:/usr/local/cuda/lib64:/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/lib/nvidia:${LD_LIBRARY_PATH}"

ENV MUJOCO_GL=egl
ENV PYOPENGL_PLATFORM=egl
ENV XLA_PYTHON_CLIENT_PREALLOCATE=false

RUN git clone --recursive https://github.com/nakamotoo/mj_envs /opt/mj_envs \
    && git -C /opt/mj_envs submodule update --init --recursive

# Pre-install the locked third-party dependencies into the image's venv so
# containers start with them ready. Enable with
# `docker build --build-arg PRIME_DEPS=1 .`. The project's own packages are not
# installed here; entrypoint.sh installs them from the mounted checkout.
ARG PRIME_DEPS=0
COPY pyproject.toml uv.lock /opt/uvprime/
COPY robomimic/pyproject.toml /opt/uvprime/robomimic/pyproject.toml
RUN if [ "$PRIME_DEPS" = "1" ]; then \
      cd /opt/uvprime && \
      uv sync --frozen --no-install-workspace --no-install-project && \
      rm -rf /root/.cache/uv ; \
    fi

COPY entrypoint.sh /entrypoint.sh

WORKDIR /workspace
ENTRYPOINT ["/entrypoint.sh"]
