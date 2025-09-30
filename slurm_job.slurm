#!/bin/sh
#SBATCH --account=studiegrupper-cogito
#SBATCH --job-name="ksim-zbot-train"
#SBATCH --time=06:00:00
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:1
#SBATCH --mem=40GB
#SBATCH --nodes=1
#SBATCH --constraint="(h100|h200)&gpu80g"
#SBATCH --output=slurm_outputs/ksim_output.txt
#SBATCH --error=slurm_outputs/ksim_error.txt
#SBATCH --mail-user=sebasrb@ntnu.no
#SBATCH --mail-type=ALL

WORKDIR=${SLURM_SUBMIT_DIR}
cd ${WORKDIR}
echo "Running from directory: $SLURM_SUBMIT_DIR"
echo "Job name: $SLURM_JOB_NAME"
echo "Job ID: $SLURM_JOB_ID"
echo "Node(s): $SLURM_JOB_NODELIST"

# Create output directory if it doesn't exist
mkdir -p slurm_outputs

module purge
module load Python/3.11.3-GCCcore-12.3.0
module load CUDA/12.1.1

# Install dependencies with compatible JAX version (4 months old)
pip install -r requirements.txt
pip install 'jax[cuda12]'

# Verify GPU backend

# Set MuJoCo to use headless rendering backend (osmesa)
export MUJOCO_GL=osmesa

# Run training
python -m train max_steps=100

# Example: to load pretrained checkpoint and run in view mode, uncomment:
# python -m train load_from_ckpt_path=assets/ckpt.bin run_mode=view

# No conda to deactivate

echo "Job finished"
