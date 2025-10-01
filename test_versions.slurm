#!/bin/sh
#SBATCH --account=studiegrupper-cogito
#SBATCH --job-name="ksim-version-test"
#SBATCH --time=00:10:00
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:1
#SBATCH --mem=8GB
#SBATCH --nodes=1
#SBATCH --output=slurm_outputs/version_test_output.txt
#SBATCH --error=slurm_outputs/version_test_error.txt
#SBATCH --mail-user=sebasrb@ntnu.no
#SBATCH --mail-type=ALL

WORKDIR=${SLURM_SUBMIT_DIR}
cd ${WORKDIR}
echo "Running version test from directory: $SLURM_SUBMIT_DIR"

# Create output directory if it doesn't exist
mkdir -p slurm_outputs

# Path to your conda env
ENV_PATH="/cluster/work/$USER/ksim-gym-example/ksim-env/"

module purge
module load Anaconda3/2024.02-1

# Activate environment
source activate ${ENV_PATH}

echo "Python version:"
python --version

echo "MuJoCo version:"
python -c "import mujoco; print('MuJoCo version:', mujoco.__version__)"

echo "JAX version:"
python -c "import jax; print('JAX version:', jax.__version__)"

echo "JAX backend:"
python -c "import jax; print('JAX backend:', jax.default_backend())"

echo "Testing training pipeline step by step:"
python -c "
import sys
import traceback

try:
    print('1. Importing modules...')
    import train
    import ksim
    import asyncio

    print('2. Creating task with minimal config...')
    # Create a minimal config object
    import argparse
    from dataclasses import dataclass

    @dataclass
    class MinimalConfig:
        max_steps: int = 10

    config = MinimalConfig()

    print('3. Creating HumanoidWalkingTask...')
    task = train.HumanoidWalkingTask(config=config)

    print('4. Getting MuJoCo model...')
    mj_model = task.get_mujoco_model()
    print(f'   Model loaded with {mj_model.njnt} joints')

    print('5. Getting physics model...')
    physics_model = task.get_physics_model(mj_model)
    print('   Physics model created')

    print('6. Testing events and resets (our tuple fix)...')
    events = task.get_events(physics_model)
    resets = task.get_resets(physics_model)
    print(f'   Events type: {type(events)}, length: {len(events)}')
    print(f'   Resets type: {type(resets)}, length: {len(resets)}')

    print('7. Creating engine...')
    engine = ksim.engine.MjxEngine(
        physics_model=physics_model,
        resets=resets,
        events=events,
    )
    print('   Engine created successfully')

    print('8. Testing engine reset...')
    # This is where the error likely occurs
    import jax
    rng = jax.random.PRNGKey(0)
    state = engine.reset(physics_model, 0.5, rng)
    print('   Engine reset successful!')

except Exception as e:
    print(f'ERROR at step: {str(e)}')
    print(f'Error type: {type(e).__name__}')
    print('Full traceback:')
    traceback.print_exc()
"

conda deactivate
echo "Version test finished"