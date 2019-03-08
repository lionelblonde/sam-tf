#!/usr/bin/env bash
# Example: ./mujoco_gather_xpo_expert.sh <env_id> <xpo_pol_ckpt_dir_path> <num_trajs>

cd ../..

python -m imitation.expert_algorithms.run_xpo_expert \
    --note="" \
    --env_id=$1 \
    --no-from_raw_pixels \
    --seed=0 \
    --log_dir="data/logs" \
    --task="gather_xpo_expert" \
    --model_ckpt_dir=$2 \
    --demos_dir="data/expert_demonstrations" \
    --hid_widths 64 64 \
    --hid_nonlin="leaky_relu" \
    --num_trajs=$3 \
    --no-sample_or_mode \
    --no-render
