import os
import torch
# os.environ["CUDA_VISIBLE_DEVICES"]="3"
from transformers import AutoTokenizer,AutoModelForMaskedLM
import argparse
import logging
parser = argparse.ArgumentParser()
parser.add_argument('--rewrite_path',type=str,default="/lustre/gst/xuchunfu/zhangxt/TMPNN/soluble/real/esm_denovo.fa", help="name")
parser.add_argument('--model',type=str,default="facebook/esm2_t48_15B_UR50D", help="learning rate of Adam optimizer")

args = parser.parse_args()

tokenizer = AutoTokenizer.from_pretrained(args.model)
model = AutoModelForMaskedLM.from_pretrained(args.model)
model=model.cuda()

softmax=torch.nn.Softmax(dim=-1)
with open(args.rewrite_path, 'r') as f:
    seqs = f.readlines()
p_sum = 0
print('starting predict')
model.eval()


with open(args.rewrite_path, 'a+')as f:
    for i, seq in enumerate(seqs):
        if not seq.startswith('>'):
            with torch.no_grad():
                seq = seq[:-1] # 去掉换行符
                inputs = tokenizer(seq, return_tensors="pt")
                for key in inputs:
                        inputs[key] = inputs[key].cuda()
                logits = model(**inputs).logits[0, ...]
                p = softmax(logits)[1:-1]
                indices=inputs["input_ids"].permute(1,0)[1:-1]
                selected_elements = torch.gather(p, dim=1, index=indices)
                p_mean = torch.mean(selected_elements)
                title = seqs[i-1][:-1] 
                seqs[i-1] = title[:title.rfind(',')] + ',' + str(round(float(p_mean), 4)) + '\n'
    f.truncate(0)
    for seq in seqs:       
        f.write(seq)
print('finish')
