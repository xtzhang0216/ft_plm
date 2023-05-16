#!/bin/bash
#SBATCH --job-name=evo
#SBATCH -p gpu4
#SBATCH --gres=gpu:1
#SBATCH --output=%j_evolution.out
#SBATCH --error=%j.err
source activate
conda activate protein_predict
for kind in soluble; do 
    for design in original_design; do
        for fa in /lustre/gst/xuchunfu/zhangxt/TMPNN/$kind/$design/*
            do
                python3 /lustre/gst/xuchunfu/zhangxt/ft_plm/compute_log_15B.py \
                    --read_path $fa \
                    --write_path /lustre/gst/xuchunfu/zhangxt/TMPNN/soluble/15b_o/$(basename $fa) \
                    --model /lustre/gst/xuchunfu/zhangxt/.cache/huggingface/hub/models--facebook--esm2_t48_15B_UR50D/snapshots/5fbca39631164edc1d402a5aa369f982f72ee282/
        done
    done
done