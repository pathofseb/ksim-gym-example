# How to Switch Robot Models in K-Sim Gym

This tutorial explains how to replace the default "kbot" robot model with a different URDF/robot model in your K-Sim training setup.

## Overview

The K-Sim gym uses robot models through the `ksim` library, which loads robot configurations from a centralized model repository. The robot model is specified in the main training code and affects:

- Physical simulation parameters
- Joint names and configurations
- Actuator specifications
- Reset positions and limits
- Reward functions

## Key Files to Modify

### 1. Main Training Script (`train.py`)

The robot model is loaded in the `HumanoidWalkingTask` class:

```python
def get_mujoco_model(self) -> mujoco.MjModel:
    mjcf_path = asyncio.run(ksim.get_mujoco_model_path("kbot", name="robot"))
    return mujoco_scenes.mjcf.load_mjmodel(mjcf_path, scene="smooth")

def get_mujoco_model_metadata(self, mj_model: mujoco.MjModel) -> ksim.Metadata:
    metadata = asyncio.run(ksim.get_mujoco_model_metadata("kbot"))
    return metadata
```

**Key Change:** Replace `"kbot"` with your desired robot model name.

### 2. Joint Configuration (`ZEROS` array)

The default joint positions are defined at the top of `train.py`:

```python
ZEROS: list[tuple[str, float]] = [
    ("dof_right_shoulder_pitch_03", 0.0),
    ("dof_right_shoulder_roll_03", math.radians(-10.0)),
    ("dof_right_elbow_02", math.radians(90.0)),
    # ... more joint definitions
]
```

**Key Change:** Update joint names and default positions to match your robot model.

## Step-by-Step Guide

### Step 1: Identify Available Robot Models

First, check what robot models are available in the ksim library:

```python
import ksim
import asyncio

# List available models (this may require exploring the ksim documentation)
# Common models might include: "kbot", "humanoid", "atlas", etc.
```

### Step 2: Update Robot Model Name

In `train.py`, find these two functions and replace `"kbot"` with your robot model:

```python
def get_mujoco_model(self) -> mujoco.MjModel:
    mjcf_path = asyncio.run(ksim.get_mujoco_model_path("YOUR_ROBOT_NAME", name="robot"))
    return mujoco_scenes.mjcf.load_mjmodel(mjcf_path, scene="smooth")

def get_mujoco_model_metadata(self, mj_model: mujoco.MjModel) -> ksim.Metadata:
    metadata = asyncio.run(ksim.get_mujoco_model_metadata("YOUR_ROBOT_NAME"))
    return metadata
```

### Step 3: Update Joint Names and Default Positions

1. **Get Joint Names:** After loading your new robot model, inspect the joint names:

```python
import mujoco
model = task.get_mujoco_model()
joint_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
               for i in range(model.njnt)]
print("Available joints:", joint_names)
```

2. **Update ZEROS Array:** Replace the `ZEROS` array with joints from your robot:

```python
ZEROS: list[tuple[str, float]] = [
    ("joint_name_1", 0.0),
    ("joint_name_2", math.radians(10.0)),
    # Add all joints from your robot with appropriate default positions
]
```

### Step 4: Update Reward Functions (Optional)

Some reward functions may be robot-specific. Check these functions in the `HumanoidWalkingTask` class:

- `BentArmPenalty.create_penalty()` - may reference specific arm joints
- `StraightLegPenalty.create_penalty()` - may reference specific leg joints

Update joint references if needed:

```python
def get_rewards(self, physics_model: ksim.PhysicsModel) -> list[ksim.Reward]:
    return [
        # Update these if they reference specific joint names
        BentArmPenalty.create_penalty(physics_model, scale=-0.1),
        StraightLegPenalty.create_penalty(physics_model, scale=-0.1),
    ]
```

### Step 5: Update Observations (If Needed)

Check if any observations reference specific body parts or joints:

```python
def get_observations(self, physics_model: ksim.PhysicsModel) -> list[ksim.Observation]:
    return [
        ksim.JointPositions.create(physics_model=physics_model),
        ksim.JointVelocities.create(physics_model=physics_model),
        # These may need updating for different robot morphologies:
        ksim.BodyPosition.create(
            physics_model=physics_model,
            body="torso",  # Update body name if different
        ),
        # ... other observations
    ]
```

### Step 6: Test the Changes

1. **Dry Run:** First test that the model loads without errors:

```python
python -c "
from train import HumanoidWalkingTask
task = HumanoidWalkingTask()
model = task.get_mujoco_model()
print('Model loaded successfully!')
print('Number of joints:', model.njnt)
"
```

2. **Training Test:** Run a short training session:

```bash
python -m train max_steps=10
```

## Common Issues and Solutions

### Issue 1: Joint Names Don't Match

**Error:** KeyError or "joint not found" errors

**Solution:**
1. Print all available joints as shown in Step 3
2. Update the `ZEROS` array with correct joint names
3. Remove any joints that don't exist in the new model

### Issue 2: Different Robot Morphology

**Error:** Reward functions fail or give unexpected results

**Solution:**
1. Comment out robot-specific penalties initially
2. Use generic rewards like `ksim.NaiveForwardReward`
3. Gradually add back penalties after updating joint references

### Issue 3: Model Not Found

**Error:** "Model 'your_robot' not found"

**Solution:**
1. Check ksim documentation for available models
2. Ensure the model name is spelled correctly
3. You may need to add custom URDF files to the ksim model repository

## Example: Switching from kbot to hypothetical "atlas" robot

```python
# In train.py

# 1. Update model loading
def get_mujoco_model(self) -> mujoco.MjModel:
    mjcf_path = asyncio.run(ksim.get_mujoco_model_path("atlas", name="robot"))
    return mujoco_scenes.mjcf.load_mjmodel(mjcf_path, scene="smooth")

def get_mujoco_model_metadata(self, mj_model: mujoco.MjModel) -> ksim.Metadata:
    metadata = asyncio.run(ksim.get_mujoco_model_metadata("atlas"))
    return metadata

# 2. Update joint configuration (example)
ZEROS: list[tuple[str, float]] = [
    ("r_arm_shz", 0.0),
    ("r_arm_shx", math.radians(-10.0)),
    ("r_arm_ely", math.radians(90.0)),
    ("l_arm_shz", 0.0),
    ("l_arm_shx", math.radians(10.0)),
    ("l_arm_ely", math.radians(-90.0)),
    # ... continue with all atlas joints
]
```

## Notes

- Always backup your working configuration before making changes
- Test incrementally - start with model loading, then add complexity
- Some robots may require different scene configurations (`scene="smooth"` parameter)
- Joint limits and ranges may be different, requiring tuning of reset scales and penalty parameters
- The training hyperparameters may need adjustment for different robot morphologies

## Getting Joint Information

To discover joint names and properties for your robot:

```python
import mujoco
import asyncio
import ksim

# Load your robot model
mjcf_path = asyncio.run(ksim.get_mujoco_model_path("YOUR_ROBOT", name="robot"))
model = mujoco.MjModel.from_xml_path(mjcf_path)

# Print joint information
print("Joint Names and Ranges:")
for i in range(model.njnt):
    joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
    joint_range = model.jnt_range[i]
    print(f"  {joint_name}: range = {joint_range}")
```

This tutorial should help you successfully switch robot models in your K-Sim training setup!