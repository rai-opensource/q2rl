#!/bin/bash
# Run from project root.

# Add project root to PYTHONPATH
PROJECT_ROOT="$(git rev-parse --show-toplevel)"
export PYTHONPATH=$PROJECT_ROOT:$PYTHONPATH

# For adroit mj_envs in case editable install didn't work
export PYTHONPATH="$PROJECT_ROOT/mj_envs:$PYTHONPATH"

# If logs/ directory doesn't exist, create it
if [ ! -d "$PROJECT_ROOT/logs" ]; then
    mkdir -p "$PROJECT_ROOT/logs"
fi

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYOPENGL_PLATFORM=egl
export MUJOCO_GL=egl
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia

# we can use same config as wsrl
seed=(0 10 20 30 40)
for i in {0..4}
do
    python3 main/ibrl/ibrl_gym.py \
    --agent ibrl_state \
    --config experiments/configs/train_config.py:adroit_wsrl \
    --group door_ibrl \
    --reward_scale 10.0 \
    --reward_bias 5.0 \
    --env door-binary-v0 \
    --resume_path_bc $PROJECT_ROOT/data/policies/bc_policies/door/checkpoint_100000 \
    --utd 4 \
    --seed ${seed[$i]} \
    --batch_size 256 \
    --num_online_steps 300_000 \
    --eval_interval 20_000 \
    --log_interval 5000 \
    --utd 4 &> logs/door_ibrl_${seed[$i]}.out &
done
wait