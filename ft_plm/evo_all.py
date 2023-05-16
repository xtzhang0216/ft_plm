import os
import torch
# os.environ["CUDA_VISIBLE_DEVICES"]="3"
from transformers import AutoTokenizer,AutoModelForMaskedLM
import argparse
import logging
parser = argparse.ArgumentParser()
parser.add_argument('--write_path',type=str,default="/pubhome/xtzhang/data/seq_for_zb/from_zxt/design_2/6eid_renum.fa", help="name")
parser.add_argument('--read_path',type=str,default="/pubhome/bozhang/TMPNN/TMPNN_beta/TMPNN_beta_ipa/for_zhangbo/design_2/6eid_renum.fa", help="learning rate of Adam optimizer")
parser.add_argument('--model',type=str,default="facebook/esm2_t48_15B_UR50D", help="learning rate of Adam optimizer")

args = parser.parse_args()
write_path = args.write_path
read_path = args.read_path
name = read_path[-7:-3]
write_path = write_path + name + '.fa'

def evo_all(args, read_path, write_path):
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForMaskedLM.from_pretrained(args.model)
    model=model.cuda()

    softmax=torch.nn.Softmax(dim=-1)
    with open(read_path, 'r') as f:
        seqs = f.readlines()
    p_sum = 0
    print('starting predict')
    model.eval()

    with open(write_path, 'a+')as f:
        for i, seq in enumerate(seqs):
            if not seq.startswith('>'):
                with torch.no_grad():
                    seq = seq[:-1] # 去掉换行符
                    inputs = tokenizer(seq, return_tensors="pt")
                    for key in inputs:
                            inputs[key] = inputs[key].cuda()
                    logits = model(**inputs).logits[0, ...]
                    p = softmax(logits)[1:-1]
                    # indices=inputs["input_ids"].permute(1,0)
                    max_indices = torch.argmax(p, dim=1)
                    selected_elements = torch.gather(p, dim=1, index=max_indices[:,None])
                    p_mean = torch.mean(selected_elements)
                    seqs[i] = ''.join(tokenizer.convert_ids_to_tokens(max_indices)) + '\n'
                    title = seqs[i-1][:-1] 
                    seqs[i-1] = title[:title.rfind(',')] + ',' + str(round(float(p_mean), 4)) + '\n'
        f.truncate(0)
        for seq in seqs:       
            f.write(seq)
    print('finish')

if os.path.exists(write_path):
    print('jump')
else:
    evo_all(args, read_path, write_path)