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
    python3 main/q2rl/q2rl_gym.py \
    --agent q2rl  \
    --config experiments/configs/train_config.py:kitchen_wsrl \
    --group kitchen_q2rl \
    --reward_scale 1.0 \
    --reward_bias -4.0 \
    --env kitchen-complete-v0 \
    --resume_path_bc $PROJECT_ROOT/data/policies/bc_policies/kitchen/checkpoint_200000 \
    --utd 4 \
    --seed ${seed[$i]} \
    --batch_size 1024 \
    --bc_weight 0.3 \
    --get_new_rollouts=True \
    --num_eval_rollouts=50 \
    --replay_path=rollouts.pkl \
    --num_q_est_steps 20000 \
    --num_online_steps 500_000 \
    --eval_interval 20_000 \
    --video_interval 60_000 \
    --log_interval 10_000 \
    --utd 4 &> logs/kitchen_q2rl_${seed[$i]}.out &
done
wait