# Containerized replication path for the Clay v1.5 paper. See REPRODUCING.md ("Replication in a container").
#   docker build -t clay-v15-paper .
#   docker run --gpus all clay-v15-paper make repro-smoke
# The torch 2.14.1 wheels bundle CUDA 13.0, so a plain Python base image is enough; the host only needs an
# NVIDIA driver (>= 580) and the NVIDIA container toolkit.
FROM python:3.12-slim-bookworm

RUN apt-get update \
 && apt-get install -y --no-install-recommends make git curl ca-certificates build-essential libgl1 libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/* \
 && pip install --no-cache-dir uv==0.13.5

WORKDIR /repo
# Dependency layers first, so editing a script does not reinstall the environments.
COPY pyproject.toml uv.lock Makefile ./
RUN make env \
 && UV_PROJECT_ENVIRONMENT=/repo/.venv-olmo uv sync --frozen --no-default-groups --group olmo
COPY . /repo
RUN make env

ENV HF_HOME=/cache/hf \
    CLAY_SCRATCH=/cache/scratch \
    CKPT=/cache/checkpoints/v1.5/clay-v1.5.ckpt \
    UV_LINK_MODE=copy
