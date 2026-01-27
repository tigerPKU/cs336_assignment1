#!/bin/bash
export PYTHONPATH=$PYTHONPATH:.

# 创建日志目录
mkdir -p logs/lr_sweep
mkdir -p checkpoints_experiments

TRAIN_DATA="data_preprocessed/tinystories_train.npy"
VAL_DATA="data_preprocessed/tinystories_train.npy"
OUT_DIR="checkpoints_experiments"
WANDB_PROJECT="cs336-assignment1-lr-sweep"

echo "=== Starting Learning Rate Sweep ==="

# 定义要测试的学习率列表
# 1e-2: 极大，通常会由梯度爆炸导致 Loss 变成 NaN (Divergence)
# 5e-3: 很大，处于边缘
# 1e-3: 较大
# 6e-4: 默认值 (通常比较稳)
# 1e-4: 保守值
LEARNING_RATES=("1e-2" "5e-3" "1e-3" "6e-4" "1e-4")

for lr in "${LEARNING_RATES[@]}"; do
    echo "------------------------------------------------"
    echo "Running Experiment with LR = $lr"
    echo "------------------------------------------------"
    
    # Run Name 格式: lr_1e-2, lr_6e-4 等
    RUN_NAME="lr_${lr}"
    
    python scripts/train.py \
        --train_path $TRAIN_DATA \
        --val_path $VAL_DATA \
        --out_dir $OUT_DIR \
        --device cuda \
        --wandb_project $WANDB_PROJECT \
        --wandb_run_name $RUN_NAME \
        --lr $lr \
        --max_iters 5000 \
        > logs/lr_sweep/${RUN_NAME}.txt 2>&1
        
    echo "Finished LR=$lr. Log saved to logs/lr_sweep/${RUN_NAME}.txt"
done

echo "=== LR Sweep Finished ==="