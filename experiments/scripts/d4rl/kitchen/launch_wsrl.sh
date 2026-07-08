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
    python3 main/offline_to_online_rl/finetune_gym.py \
    --agent sac \
    --config experiments/configs/train_config.py:kitchen_wsrl \
    --group kitchen_wsrl \
    --num_offline_steps 250_000 \
    --resume_path $PROJECT_ROOT/data/policies/rl_offline/kitchen/calql/checkpoint_250000 \
    --reward_scale 1.0 \
    --reward_bias -4.0 \
    --env kitchen-complete-v0 \
    --utd 4 \
    --batch_size 1024 \
    --seed ${seed[$i]} \
    --warmup_steps 5000  &> logs/wsrl_kitchen_${seed[$i]}.out &
done
wait