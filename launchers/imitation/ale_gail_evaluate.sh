#!/usr/bin/env bash
# Example: ./ale_gail_evaluate.sh <env_id> <gail_pol_ckpt_dir_path> <num_trajs>

cd ../..

python -m imitation.imitation_algorithms.run_gail \
    --note="" \
    --env_id=$1 \
    --from_raw_pixels \
    --seed=0 \
    --log_dir="data/logs" \
    --task="evaluate_gail_policy" \
    --pol_nums_filters 8 16 \
    --pol_filter_shapes 8 4 \
    --pol_stride_shapes 4 2 \
    --pol_hid_widths 128 \
    --d_nums_filters 8 16 \
    --d_filter_shapes 8 4 \
    --d_stride_shapes 4 2 \
    --d_hid_widths 128 \
    --hid_nonlin="tanh" \
    --num_trajs=$3 \
    --sample_or_mode \
    --no-render \
    --record \
    --video_dir="data/videos" \
    --model_ckpt_dir=$2
