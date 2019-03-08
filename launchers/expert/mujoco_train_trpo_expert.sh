#!/usr/bin/env bash
# Example: ./mujoco_train_trpo_expert.sh <num_mpi_workers> <env_id>

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
    --algo='trpo' \
    --rmsify_obs \
    --save_frequency=10 \
    --num_iters=1000000 \
    --timesteps_per_batch=1024 \
    --batch_size=64 \
    --sample_or_mode \
    --hid_widths 64 64 \
    --hid_nonlin="leaky_relu" \
    --hid_w_init="he_normal" \
    --gaussian_fixed_var \
    --no-with_layernorm \
    --max_kl=0.01 \
    --ent_reg_scale=0. \
    --cg_iters=10 \
    --cg_damping=0.1 \
    --vf_iters=5 \
    --vf_lr=1e-3 \
    --gamma=0.99 \
    --gae_lambda=0.98
