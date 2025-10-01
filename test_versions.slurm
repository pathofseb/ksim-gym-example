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

echo "Testing MuJoCo-JAX compatibility:"
python -c "
import mujoco
import mujoco.mjx as mjx
import jax

# Test basic MuJoCo model creation
print('Testing basic MuJoCo model...')
xml = '''
<mujoco>
  <worldbody>
    <body>
      <geom type=\"sphere\" size=\"1\"/>
    </body>
  </worldbody>
</mujoco>
'''
model = mujoco.MjModel.from_xml_string(xml)
data = mujoco.MjData(model)
print('Basic MuJoCo model created successfully')

# Test MJX conversion
print('Testing MJX conversion...')
try:
    mjx_model = mjx.put_model(model)
    mjx_data = mjx.put_data(model, data)
    print('MJX conversion successful')
    print('MJX data fields:', dir(mjx_data))
except Exception as e:
    print('MJX conversion failed:', str(e))
    print('Error type:', type(e).__name__)
"

conda deactivate
echo "Version test finished"