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

seed=(0 10 20 30 40)
for i in {0..4}
do
    python main/offline_to_online_rl/finetune_gym.py \
    --agent calql \
    --config experiments/configs/train_config.py:adroit_cql \
    --group pen_calql \
    --warmup_steps 0 \
    --num_offline_steps 20_000 \
    --reward_scale 10.0 \
    --reward_bias 5.0 \
    --seed ${seed[$i]} \
    --env pen-binary-v0 &> logs/pen_calql_finetune_${seed[$i]}.out &
done
wait