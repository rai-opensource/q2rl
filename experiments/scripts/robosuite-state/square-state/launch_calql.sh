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
    python main/offline_to_online_rl/finetune_robosuite.py \
    --agent calql \
    --config experiments/configs/train_config.py:robosuite_cql \
    --group square_state_calql \
    --env square \
    --warmup_steps 0 \
    --num_offline_steps 500_000 \
    --num_online_steps 500_001 \
    --reward_scale 5.0 \
    --reward_bias -1.0 \
    --eval_interval=20_000 \
    --video_interval=50_000 \
    --save_interval=50_000 \
    --log_interval=10000 \
    --resume_path_bc=data/policies/bc_policies/square/gmm.pth \
    --render=True \
    --get_demo_buffer=True \
    --batch_size=256 \
    --data_filter_key=train \
    --demo_path data/datasets/square/ph/low_dim_v141.hdf5 \
    --seed ${seed[$i]} &> logs/square_state_calql_${seed[$i]}.out &
done
wait