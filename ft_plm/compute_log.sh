#!/bin/bash
#SBATCH --job-name=evo
#SBATCH -p gpu4
#SBATCH --gres=gpu:1
#SBATCH --output=%j_evolution.out
#SBATCH --error=%j.err
conda activate protein_predict


python3 /lustre/gst/xuchunfu/zhangxt/ft_plm/compute_log.py \
                    --rewrite_path /lustre/gst/xuchunfu/zhangxt/TMPNN/soluble/real/esm_denovo.fa \
                    --model /lustre/gst/xuchunfu/zhangxt/.cache/huggingface/hub/models--facebook--esm2_t48_15B_UR50D/snapshots/5fbca39631164edc1d402a5aa369f982f72ee282/


