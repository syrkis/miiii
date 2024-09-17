# Default recipe to run when just is called without arguments
default:
    @just --list

# Run the Python script with Poetry
run *ARGS:
    poetry run python main.py {{ARGS}}
    ln -sf $(shell ls -td paper/figs/runs/* | head -n 1) paper/figs/latest

# Install dependencies
install:
    poetry install

# Update dependencies
update:
    poetry update
