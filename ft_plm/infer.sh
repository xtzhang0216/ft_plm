#!/bin/bash
#SBATCH --job-name=getesm
#SBATCH -p gpu2
#SBATCH --gres=gpu:1
#SBATCH --output=%j.out
#SBATCH --error=%j.err

source activate
conda activate protein_predict

python /lustre/gst/xuchunfu/zhangxt/ft_plm/eva_token.py \
    --model mlmp00 \
    --parameters facebook/esm2_t36_3B_UR50D
    

python /lustre/gst/xuchunfu/zhangxt/ft_plm/eva_token.py \
    --model mlmp15 \
    --parameters /lustre/gst/xuchunfu/zhangxt/checkpoint/230330_token/checkpoint-6447/

python /lustre/gst/xuchunfu/zhangxt/ft_plm/eva_token.py \
    --model mlmp30 \
    --parameters /lustre/gst/xuchunfu/zhangxt/checkpoint/230508_token_mlp3/checkpoint-12894/

python /lustre/gst/xuchunfu/zhangxt/ft_plm/eva_token.py \
    --model mlmp45 \
    --parameters /lustre/gst/xuchunfu/zhangxt/checkpoint/230508_token_mlp45/checkpoint-20262/

python /lustre/gst/xuchunfu/zhangxt/ft_plm/eva_token.py \
    --model mlmp50 \
    --parameters /lustre/gst/xuchunfu/zhangxt/checkpoint/230503_token_mlp5/checkpoint-9210/

python /lustre/gst/xuchunfu/zhangxt/ft_plm/eva_token.py \
    --model mlmp60 \
    --parameters /lustre/gst/xuchunfu/zhangxt/checkpoint/230508_token_mlp6/checkpoint-31314/

python /lustre/gst/xuchunfu/zhangxt/ft_plm/eva_token.py \
    --model mlmp70 \
    --parameters /lustre/gst/xuchunfu/zhangxt/checkpoint/230506_token_mlp7/checkpoint-27630/


# python /lustre/gst/xuchunfu/zhangxt/ft_plm/eva_token.py \
    # --model "0000" \
    # --parameters /lustre/gst/xuchunfu/zhangxt/.cache/torch/hub/checkpoints/esm2_t36_3B_UR50D-contact-regression.pt

