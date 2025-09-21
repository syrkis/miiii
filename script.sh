#!/bin/bash
#SBATCH --output=outs/job.%j.out      # Name of output file (%j expands to jobId)
#SBATCH --gres=gpu:a100_80gb:1
#SBATCH --time=00:50:00
#SBATCH --partition=acltr
#SBATCH --nodelist=cn[7]
#SBATCH --exclusive
#SBATCH --nodes=1
# #SBATCH --gpus=1

# Run your Python script
export XLA_PYTHON_CLIENT_MEM_FRACTION="1.00"
export JAX_ENABLE_X64=true
time singularity exec --nv miiii.sif uv run python main.py \
    p=113 \
    epochs=65536 \
    d=256
