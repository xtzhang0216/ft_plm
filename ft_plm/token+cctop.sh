#!/bin/bash
#SBATCH --job-name=token
#SBATCH -p gpu1
#SBATCH --gres=gpu:1
#SBATCH --output=/lustre/gst/xuchunfu/zhangxt/ft_plm/%j.out
#SBATCH --error=/lustre/gst/xuchunfu/zhangxt/ft_plm/%j.err

source activate
conda activate protein_predict

WANDB__SERVICE_WAIT=300 python /lustre/gst/xuchunfu/zhangxt/ft_plm/ft_token+cctop.py \
    --batch_size 16 \
    --epoch 50 \
    --project_name ft_plm \
    --log_interval 200 \
    --jsonl /lustre/gst/xuchunfu/zhangxt/data/tmpnn_v8.jsonl \
    --ratio 0.2 \
    --train_mlmp 0.6 \
    --test_mlmp 0.3 \
    --run_name 230501_tokencctop_6 \
    --save_folder /lustre/gst/xuchunfu/zhangxt/checkpoint/230501_tokencctop_6/ \