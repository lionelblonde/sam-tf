#!/usr/bin/env bash
# Example: ./mujoco_sam_evaluate.sh <env_id> <sam_pol_ckpt_dir_path> <num_trajs>

cd ../..

python -m imitation.imitation_algorithms.run_sam \
    --note="" \
    --env_id=$1 \
    --no-from_raw_pixels \
    --seed=0 \
    --rmsify_obs \
    --rmsify_rets \
    --noise_type=none \
    --with_layernorm \
    --ac_branch_in=1 \
    --no-prioritized_replay \
    --no-ranked \
    --no-add_demos_to_mem \
    --no-unreal \
    --wd_scale=1e-3 \
    --n_step_returns \
    --n=96 \
    --log_dir="data/logs" \
    --task="evaluate_sam_policy" \
    --actorcritic_hid_widths 64 64 \
    --d_hid_widths 64 64 \
    --hid_nonlin="leaky_relu" \
    --num_trajs=$3 \
    --no-render \
    --record \
    --video_dir="data/videos" \
    --model_ckpt_dir=$2
