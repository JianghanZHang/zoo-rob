#!/bin/bash
# Launch full RAC benchmark overnight.
#
# Methods: rac (learned global ω), rac-fixed (geometric ω), rac-state (state-dependent ω),
#          ppo, ddpg, td3
# Envs: HalfCheetah-v4, Hopper-v4, Walker2d-v4, Ant-v4, InvertedPendulum-v4
# Seeds: 1-5
# Total timesteps: 1M each
#
# Usage: bash launch_benchmark.sh [max_parallel_jobs]

MAX_PARALLEL=${1:-4}
TOTAL_TIMESTEPS=1000000
SEEDS="1 2 3 4 5"
ENVS="HalfCheetah-v4 Hopper-v4 Walker2d-v4 Ant-v4 InvertedPendulum-v4"

# RAC variants
RAC_METHODS="rac rac-fixed rac-state"
RAC_COMMON="--total_timesteps $TOTAL_TIMESTEPS --omega_epochs 10 --critic_epochs 10 --actor_epochs 5 --N 32"

# SB3-based methods (use their own CLI)
SB3_METHODS="ppo td3"

LOG_DIR="logs/benchmark_$(date +%Y%m%d_%H%M%S)"
mkdir -p $LOG_DIR

echo "Launching benchmark to $LOG_DIR"
echo "Max parallel: $MAX_PARALLEL"
echo ""

job_count=0

# RAC variants
for method in $RAC_METHODS; do
    for env in $ENVS; do
        for seed in $SEEDS; do
            while [ $(jobs -r | wc -l) -ge $MAX_PARALLEL ]; do
                sleep 5
            done

            extra_args=""
            if [ "$method" = "rac-state" ]; then
                extra_args="--lr_omega 3e-4"
            fi

            log_file="$LOG_DIR/${method}_${env}_s${seed}.log"
            echo "Starting: $method $env seed=$seed"
            python rl/${method}.py --env_id $env --seed $seed \
                $RAC_COMMON $extra_args \
                > "$log_file" 2>&1 &

            job_count=$((job_count + 1))
        done
    done
done

# PPO and TD3 (CleanRL style, already in zoo-rob)
for method in $SB3_METHODS; do
    for env in $ENVS; do
        for seed in $SEEDS; do
            while [ $(jobs -r | wc -l) -ge $MAX_PARALLEL ]; do
                sleep 5
            done

            log_file="$LOG_DIR/${method}_${env}_s${seed}.log"
            echo "Starting: $method $env seed=$seed"
            python rl/${method}.py --env-id $env --seed $seed \
                --total-timesteps $TOTAL_TIMESTEPS \
                > "$log_file" 2>&1 &

            job_count=$((job_count + 1))
        done
    done
done

echo ""
echo "Launched $job_count jobs. Waiting for completion..."
wait
echo "All jobs complete. Results in $LOG_DIR"
