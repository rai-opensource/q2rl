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
    python3 main/ibrl/ibrl_robosuite_image.py \
        --agent=ibrl_image \
        --config=experiments/configs/train_config.py:robosuite_ibrl \
        --group=lift_img_ibrl_without_data \
        --reward_scale=5.0 \
        --reward_bias=-1.0 \
        --max_traj_length=300 \
        --env=lift \
        --resume_path_bc=data/policies/bc_policies/lift_img/gmm.pth \
        --get_demo_buffer=False \
        --render=True \
        --utd=4 \
        --seed=${seed[$i]} \
        --batch_size=256 \
        --num_online_steps=300_001 \
        --eval_interval=20_000 \
        --video_interval=20_000 \
        --save_interval=50_000 \
        --log_interval=1000  &> logs/lift_img_ibrl_without_data_${seed[$i]}.out &
done
wait