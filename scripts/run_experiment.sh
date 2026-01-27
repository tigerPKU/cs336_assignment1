#!/bin/bash
export PYTHONPATH=$PYTHONPATH:.
export WANDB_MODE=offline

# 设置通用参数
TRAIN_DATA="data_preprocessed/tinystories_train.npy"
VAL_DATA="data_preprocessed/tinystories_train.npy"
OUT_DIR="checkpoints_experiments"
WANDB_PROJECT="cs336-assignment1-experiments"
DEVICE="cuda" 

# 确保输出目录存在
mkdir -p $OUT_DIR

echo "=== Starting Section 7 Experiments ==="

# ---------------------------------------------------------
# 7.2 Problem (batch_size_experiment)
# ---------------------------------------------------------
echo "Running Batch Size Experiments..."

# Batch Size 1
python scripts/train.py --train_path $TRAIN_DATA --val_path $VAL_DATA --out_dir $OUT_DIR \
    --device $DEVICE --wandb_project $WANDB_PROJECT --wandb_run_name "batch_size_1" \
    --batch_size 1 --max_iters 5000

# Batch Size 64
python scripts/train.py --train_path $TRAIN_DATA --val_path $VAL_DATA --out_dir $OUT_DIR \
    --device $DEVICE --wandb_project $WANDB_PROJECT --wandb_run_name "batch_size_64" \
    --batch_size 64 --max_iters 5000


# ---------------------------------------------------------
# 7.3 Ablation 1: Layer Norm (Remove RMSNorm)
# ---------------------------------------------------------
echo "Running Ablation: No RMSNorm..."
python scripts/train.py --train_path $TRAIN_DATA --val_path $VAL_DATA --out_dir $OUT_DIR \
    --device $DEVICE --wandb_project $WANDB_PROJECT --wandb_run_name "ablation_no_rmsnorm" \
    --no_rms_norm \
    --lr 1e-4 


# ---------------------------------------------------------
# 7.3 Ablation 2: Pre-Norm vs Post-Norm
# ---------------------------------------------------------
echo "Running Ablation: Post-Norm..."
python scripts/train.py --train_path $TRAIN_DATA --val_path $VAL_DATA --out_dir $OUT_DIR \
    --device $DEVICE --wandb_project $WANDB_PROJECT --wandb_run_name "ablation_post_norm" \
    --norm_mode post


# ---------------------------------------------------------
# 7.3 Ablation 3: No Position Embeddings (NoPE)
# ---------------------------------------------------------
echo "Running Ablation: NoPE..."
python scripts/train.py --train_path $TRAIN_DATA --val_path $VAL_DATA --out_dir $OUT_DIR \
    --device $DEVICE --wandb_project $WANDB_PROJECT --wandb_run_name "ablation_no_rope" \
    --no_rope


# ---------------------------------------------------------
# 7.3 Ablation 4: SwiGLU vs SiLU
# ---------------------------------------------------------
echo "Running Ablation: SiLU (no gating)..."
python scripts/train.py --train_path $TRAIN_DATA --val_path $VAL_DATA --out_dir $OUT_DIR \
    --device $DEVICE --wandb_project $WANDB_PROJECT --wandb_run_name "ablation_silu" \
    --ffn_type silu --d_ff 2048


# ---------------------------------------------------------
# 7.4 OpenWebText Experiment (关键修复！)
# ---------------------------------------------------------
OWT_DATA="data_preprocessed/owt_train.npy"
if [ -f "$OWT_DATA" ]; then
    echo "Running OWT Experiment..."
    python scripts/train.py --train_path $OWT_DATA --val_path $OWT_DATA --out_dir $OUT_DIR \
        --device $DEVICE --wandb_project $WANDB_PROJECT --wandb_run_name "owt_baseline" \
        --vocab_size 50304 \
        --max_iters 5000
else
    echo "OWT data not found, skipping 7.4..."
fi

echo "All experiments finished!"