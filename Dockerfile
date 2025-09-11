FROM nvcr.io/nvidia/jax:24.10-py3

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
