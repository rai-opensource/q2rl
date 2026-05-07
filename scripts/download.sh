#!/bin/bash
set -e

# Check if hf is a command, exit if not
if ! command -v hf &> /dev/null; then
    echo "hf command not found. Please install Hugging Face CLI and authenticate before running this script."
    exit 1
fi

PROJECT_ROOT="$(git rev-parse --show-toplevel)"

# Download robomimic data
hf download --repo-type=dataset theaiinstitute/q2rl_robomimic_datasets --local-dir "$PROJECT_ROOT/data/datasets"

# Download policies
hf download --repo-type=model theaiinstitute/gmm_robomimic_can_state --local-dir "$PROJECT_ROOT/data/policies/bc_policies/can/"
hf download --repo-type=model theaiinstitute/gmm_robomimic_can_img --local-dir "$PROJECT_ROOT/data/policies/bc_policies/can_img/"
hf download --repo-type=model theaiinstitute/gmm_adroit_door_state_ckpt100k --local-dir "$PROJECT_ROOT/data/policies/bc_policies/door/checkpoint_100000/"
hf download --repo-type=model theaiinstitute/gmm_franka_kitchen_state_ckpt200k --local-dir "$PROJECT_ROOT/data/policies/bc_policies/kitchen/checkpoint_200000/"
hf download --repo-type=model theaiinstitute/gmm_robomimic_lift_state --local-dir "$PROJECT_ROOT/data/policies/bc_policies/lift/"
hf download --repo-type=model theaiinstitute/gmm_robomimic_lift_img --local-dir "$PROJECT_ROOT/data/policies/bc_policies/lift_img/"
hf download --repo-type=model theaiinstitute/gmm_adroit_pen_state_ckpt200k --local-dir "$PROJECT_ROOT/data/policies/bc_policies/pen/checkpoint_200000/"
hf download --repo-type=model theaiinstitute/gmm_robomimic_square_state --local-dir "$PROJECT_ROOT/data/policies/bc_policies/square/"
hf download --repo-type=model theaiinstitute/calql_adroit_door_state_ckpt20k --local-dir "$PROJECT_ROOT/data/policies/rl_offline/door/checkpoint_20000/"
hf download --repo-type=model theaiinstitute/calql_franka_kitchen_state_ckpt250k --local-dir "$PROJECT_ROOT/data/policies/rl_offline/kitchen/checkpoint_250000/"
hf download --repo-type=model theaiinstitute/calql_adroit_pen_state_ckpt20k --local-dir "$PROJECT_ROOT/data/policies/rl_offline/pen/checkpoint_20000/"
