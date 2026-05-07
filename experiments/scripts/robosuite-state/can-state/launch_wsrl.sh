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

export DATASET_PATH=$(python -c "from huggingface_hub import snapshot_download; print(snapshot_download('theaiinstitute/q2rl_robomimic_datasets', repo_type='dataset'))")

seed=(0 10 20 30 40)
for i in {0..4}
do
    python3 main/offline_to_online_rl/finetune_robosuite.py \
    --agent sac \
    --config experiments/configs/train_config.py:robosuite_wsrl \
    --group can_state_wsrl \
    --num_offline_steps 250_000 \
    --resume_path data/policies/rl_offline/can/calql_all_data/checkpoint_250000 \
    --reward_scale 5.0 \
    --reward_bias -1.0 \
    --eval_interval=20_000 \
    --video_interval=20_000 \
    --save_interval=50_000 \
    --log_interval=1000 \
    --resume_path_bc=data/policies/bc_policies/can/gmm.pth \
    --render=True \
    --env can \
    --utd 4 \
    --get_demo_buffer=True \
    --batch_size=256 \
    --data_filter_key=train \
    --demo_path data/datasets/can/ph/low_dim_v141.hdf5 \
    --seed ${seed[$i]} \
    --warmup_steps 5000  &> logs/can_state_wsrl_${seed[$i]}.out &
done
wait
