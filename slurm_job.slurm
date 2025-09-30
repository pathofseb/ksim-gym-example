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

# Path to your conda env (change if needed)
ENV_PATH="/cluster/work/$USER/ksim-gym-example/ksim-env/"

module purge
module load Anaconda3/2024.02-1

# Create conda environment if it doesn't exist
if [ ! -d "$ENV_PATH" ]; then
  conda create -y --prefix ${ENV_PATH} python=3.11
fi

# Activate environment
source activate ${ENV_PATH}

# Install dependencies (CPU-only JAX)
pip install -r requirements.txt
pip install jax jaxlib  # CPU-only JAX, no CUDA

echo "Installed dependencies:"
pip freeze

# Verify JAX backend
echo "Checking JAX backend:"
python -c "import jax; print('JAX version:', jax.__version__); print('JAX backend:', jax.default_backend()); print('JAX devices:', jax.devices())"

# Set MuJoCo to use headless rendering backend (osmesa)
export MUJOCO_GL=osmesa
echo "Set MUJOCO_GL=osmesa for headless rendering"

# Run training
python -m train max_steps=100

# Example: to load pretrained checkpoint and run in view mode, uncomment:
# python -m train load_from_ckpt_path=assets/ckpt.bin run_mode=view

conda deactivate

echo "Job finished"
