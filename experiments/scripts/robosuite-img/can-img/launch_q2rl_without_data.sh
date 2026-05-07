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

seed=(0 10 20)
for i in {0..1}
do
    python3 main/q2rl/q2rl_robosuite_image.py \
        --agent=q2rl_rs_image \
        --config=experiments/configs/train_config.py:robosuite_can_img \
        --group=can_img_q2rl_without_data \
        --reward_scale=5.0 \
        --reward_bias=-1.0 \
        --max_traj_length=300 \
        --env=can \
        --resume_path_bc=data/policies/bc_policies/can_img/gmm.pth \
        --get_demo_buffer=False \
        --render=True \
        --bc_weight 0.2 \
        --utd=4 \
        --seed=${seed[$i]} \
        --batch_size=256 \
        --num_eval_rollouts=100 \
        --get_new_rollouts=True \
        --replay_path=rollouts.pkl \
        --num_q_est_steps=20_000 \
        --num_online_steps=500_001 \
        --eval_interval=20_000 \
        --video_interval=20_000 \
        --save_interval=50_000 \
        --log_interval=1000 &> logs/can_img_q2rl_without_data_${seed[$i]}.out &
done
wait