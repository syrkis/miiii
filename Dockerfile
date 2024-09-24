# Start from the NVIDIA JAX container
FROM nvcr.io/nvidia/jax:23.08-py3

WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN curl -sSL https://install.python-poetry.org | python3 -

# Copy project files
COPY pyproject.toml poetry.lock* /workspace/

# Install project dependencies
RUN poetry config virtualenvs.create false \
    && poetry install --no-interaction --no-ansi

# Copy rest of project
COPY . /workspace

# Set the entry point to main script or command
CMD ["python", "main.py"]
