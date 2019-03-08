#!/usr/bin/env bash
# Example: ./mujoco_train_ppo_expert.sh <num_mpi_workers> <env_id>

cd ../..

mpirun -np $1 --allow-run-as-root python -m imitation.expert_algorithms.run_xpo_expert \
    --note="" \
    --env_id=$2 \
    --no-from_raw_pixels \
    --seed=0 \
    --checkpoint_dir="data/expert_checkpoints" \
    --summary_dir="data/summaries" \
    --log_dir="data/logs" \
    --task="train_xpo_expert" \
    --algo="ppo" \
    --rmsify_obs \
    --save_frequency=10 \
    --num_iters=1000000 \
    --timesteps_per_batch=2048 \
    --batch_size=64 \
    --optim_epochs_per_iter=10 \
    --sample_or_mode \
    --hid_widths 64 64 \
    --hid_nonlin="leaky_relu" \
    --hid_w_init="he_normal" \
    --gaussian_fixed_var \
    --no-with_layernorm \
    --ent_reg_scale=0. \
    --clipping_eps=0.2 \
    --lr=3e-4 \
    --gamma=0.99 \
    --gae_lambda=0.98 \
    --schedule="constant"
