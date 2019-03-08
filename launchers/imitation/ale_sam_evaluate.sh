#!/usr/bin/env bash
# Example: ./ale_sam_evaluate.sh <env_id> <sam_pol_ckpt_dir_path> <num_trajs>

cd ../..

python -m imitation.imitation_algorithms.run_sam \
    --note="" \
    --env_id=$1 \
    --from_raw_pixels \
    --seed=0 \
    --no-rmsify_obs \
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
    --actorcritic_nums_filters 8 16 \
    --actorcritic_filter_shapes 8 4 \
    --actorcritic_stride_shapes 4 2 \
    --actorcritic_hid_widths 128 \
    --d_nums_filters 8 16 \
    --d_filter_shapes 8 4 \
    --d_stride_shapes 4 2 \
    --d_hid_widths 128 \
    --hid_nonlin="leaky_relu" \
    --noise_type="none" \
    --num_trajs=$3 \
    --no-render \
    --record \
    --video_dir="data/videos" \
    --model_ckpt_dir=$2
