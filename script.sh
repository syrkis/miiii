#!/bin/bash
#SBATCH --output=outs/job.%j.out      # Name of output file (%j expands to jobId)
#SBATCH --time=06:00:00
# #SBATCH --gres=gpu:v100:1
#SBATCH --gpus=1

# Run your Python script
export XLA_PYTHON_CLIENT_MEM_FRACTION="1.00"
time singularity exec --nv miiii.sif uv run python main.py p=113 epochs=65536 d=256 tick=128
