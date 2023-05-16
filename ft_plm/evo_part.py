import os
import torch
# os.environ["CUDA_VISIBLE_DEVICES"]="3"
from transformers import AutoTokenizer,AutoModelForMaskedLM
import argparse
import logging
parser = argparse.ArgumentParser()
parser.add_argument('--write_path',type=str,default="/pubhome/xtzhang/data/seq_for_zb/from_zxt/design_2/6eid_renum.fa", help="name")
parser.add_argument('--read_path',type=str,default="/pubhome/bozhang/TMPNN/TMPNN_beta/TMPNN_beta_ipa/for_zhangbo/design_2/6eid_renum.fa", help="learning rate of Adam optimizer")
args = parser.parse_args()
write_path = args.write_path
read_path = args.read_path
tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t36_3B_UR50D")
model = AutoModelForMaskedLM.from_pretrained("facebook/esm2_t36_3B_UR50D")
model=model.cuda()

softmax=torch.nn.Softmax(dim=-1)
with open(read_path, 'r') as f:
    seqs = f.readlines()
update_p = 0.1
print('starting predict')
model.eval()

name = read_path[-7:-3]
write_path = write_path + name + '.fa'
with open(write_path, 'a')as f:
    for i, seq in enumerate(seqs):
        if not seq.startswith('>'):
            with torch.no_grad():
                seq = seq[:-1] # 去掉换行符
                inputs = tokenizer(seq, return_tensors="pt")
                for key in inputs:
                        inputs[key] = inputs[key].cuda()
                logits = model(**inputs).logits[0, ...]
                p = softmax(logits)
                # indices[L]
                indices=inputs["input_ids"].squeeze(0)
                # selected_elements[L,]
                selected_elements = torch.gather(p, dim=1, index=indices.unsqueeze(1)).squeeze(1)
                replace_indices = selected_elements < update_p
                max_indices = torch.argmax(p, dim=1)
                indices[replace_indices] = max_indices[replace_indices]
                seq = ''.join(tokenizer.convert_ids_to_tokens(indices)[1:-1]) + '\n'
    f.truncate(0)
    for seq in seqs:       
        f.write(seq)
print('finish')
