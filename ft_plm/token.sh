#!/bin/bash
#SBATCH --job-name=token
#SBATCH -p gpu1
#SBATCH --gres=gpu:1
#SBATCH --output=/lustre/gst/xuchunfu/zhangxt/ft_plm/230508_token_mlp75%j.out
#SBATCH --error=/lustre/gst/xuchunfu/zhangxt/ft_plm/230508_token_mlp75%j.err

source activate
conda activate protein_predict

python /lustre/gst/xuchunfu/zhangxt/ft_plm/ft_token.py \
    --run_name 230508_token_mlp75 \
    --save_folder /lustre/gst/xuchunfu/zhangxt/checkpoint/230508_token_mlp75 \
    --batch_size 16 \
    --epoch 50 \
    --project_name ft_plm \
    --log_interval 200 \
    --mlmp 0.75 \
    --jsonl /lustre/gst/xuchunfu/zhangxt/data/tmpnn_v8.jsonl \
    