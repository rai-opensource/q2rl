#!/bin/bash
# Run from project root.

# Add project root to PYTHONPATH
PROJECT_ROOT="$(git rev-parse --show-toplevel)"
export PYTHONPATH=$PROJECT_ROOT:$PYTHONPATH

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
    python3 main/ibrl/ibrl_robosuite.py \
    --agent ibrl_state \
    --config experiments/configs/train_config.py:robosuite_ibrl \
    --project q2rl \
    --group lift_state_ibrl_with_data \
    --get_demo_buffer=True \
    --data_filter_key=20_percent_train \
    --demo_path data/datasets/lift/ph/low_dim_v141.hdf5 \
    --reward_scale 5.0 \
    --reward_bias -1.0 \
    --max_traj_length=300 \
    --env lift \
    --resume_path_bc=data/policies/bc_policies/lift/gmm.pth \
    --render=True \
    --utd 4 \
    --seed ${seed[$i]} \
    --batch_size 256 \
    --num_online_steps 500_000 \
    --eval_interval 20_000 \
    --log_interval 5000 &> logs/lift_state_ibrl_with_data_${seed[$i]}.out &
done
wait