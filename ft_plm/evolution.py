import os
import torch
# os.environ["CUDA_VISIBLE_DEVICES"]="3"
from transformers import AutoTokenizer,AutoModelForMaskedLM
tokenizer = AutoTokenizer.from_pretrained("/lustre/gst/xuchunfu/zhangxt/checkpoint/token_try_a/checkpoint-4605")
model = AutoModelForMaskedLM.from_pretrained("/lustre/gst/xuchunfu/zhangxt/checkpoint/token_try_a/checkpoint-6447")
# tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t36_3B_UR50D")
# model = AutoModelForMaskedLM.from_pretrained("facebook/esm2_t36_3B_UR50D")
model=model.cuda()
path = "/lustre/gst/xuchunfu/zhangxt/data/random_set.fa"
softmax=torch.nn.Softmax(dim=-1)
with open(path, 'r') as f:
    seqs = f.readlines()
p_sum = 0
model.eval()
with torch.no_grad():
    for seq in seqs:
        if not seq.startswith('>'):
            inputs = tokenizer(seq, return_tensors="pt")
            for key in inputs:
                    inputs[key] = inputs[key].cuda()
            # [1,L,33]
            logits = model(**inputs).logits[0, ...]
            p = softmax(logits)
            indices=inputs["input_ids"].permute(1,0)
            # indices [L,1]
            selected_elements = torch.gather(p, dim=1, index=indices)
            p_mean = torch.mean(selected_elements.squeeze(-1)[1:-1])
            p_sum += p_mean
print(path)
print(2*p_sum/len(seqs))