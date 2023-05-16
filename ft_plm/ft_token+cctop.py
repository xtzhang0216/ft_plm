import torch
import json
from datasets import Dataset
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
from datasets import Dataset
from transformers import AutoModelForMaskedLM
from transformers.data.data_collator import DataCollatorForCCTOPModeling
import wandb
from torch.utils.data import DataLoader
import argparse
import torch.nn as nn
import math
import wandb
import torch.optim as optim
import os
from torch.utils.data import DataLoader
import logging
# 设置logging级别
logging.basicConfig(level=logging.INFO)
parser = argparse.ArgumentParser()
parser.add_argument('--epoch',type=int,default=10,help="epochs to train the model")
parser.add_argument('--batch_size',type=int,default=16,help="batch size tokens")
parser.add_argument('--weight_decay',type=float,default=0.01, help="weight_decay")
parser.add_argument('--jsonl',type=str,default="/lustre/gst/xuchunfu/zhangxt/data/cctop.jsonl", help="weight_decay")
parser.add_argument('--project_name',type=str,default="aaa", help="name")
parser.add_argument('--run_name',type=str,default="aaa", help="name")
parser.add_argument('--watch_freq',type=int,default=500, help="watch")
parser.add_argument('--lr',type=float,default=5e-4, help="learning rate of Adam optimizer")
parser.add_argument('--log_interval',type=int,default=2, help="learning rate of Adam optimizer")
parser.add_argument('--save_folder',type=str,default="/lustre/gst/xuchunfu/zhangxt/plm/0/", help="learning rate of Adam optimizer")
parser.add_argument('--ratio',type=float,default=0.2, help="learning rate of Adam optimizer")
parser.add_argument('--train_mlmp',type=float,default=0.3, help="train_mlmp")
parser.add_argument('--test_mlmp',type=float,default=0.3, help="test_mlmp")
parser.add_argument('--accumulation_steps',type=int,default=1, help="watch")

args = parser.parse_args()
device = torch.device('cuda')
log_interval = args.log_interval
model_checkpoint = "facebook/esm2_t36_3B_UR50D"
model_name = model_checkpoint.split("/")[-1]
num_labels, num_tokens = 6, 33
ratio = args.ratio
def load_jsonl(json_file: str) -> list:
    seq, cctop = [], []
    cctop_code = 'IMOULS'

    with open(json_file, "r") as f:
        for line in f:

                # 每一行代表一个序列的字典字符串加一个'\n',对于字符串应该用json.loads()文件进行读取
                data = json.loads(line.replace("\n", ""))
                seq.append(data["seq"])
                cctop.append([cctop_code.index(i) for i in data["cctop"]])
        for c in cctop:
            c.insert(0,-100)
            c.append(-100)
    return seq, cctop

def compute_accurracy(label, logits):
    pred = torch.argmax(logits, dim=-1)
    valid_indices = label != -100
    masked_pred = pred[valid_indices]
    masked_label = label[valid_indices]
    correct = (masked_pred == masked_label).sum().item()
    num = masked_pred.numel()
    return correct/num
seq, cctop = load_jsonl(args.jsonl)

train_sequences, test_sequences, train_cctops, test_cctops = train_test_split(seq, cctop, test_size=0.1, shuffle=True)
#需要解决加开始字符后与标签不匹配的问题，能否从list生产dataset类？
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
train_tokenized = tokenizer(train_sequences)
test_tokenized = tokenizer(test_sequences)
# tokenizer.pad(train_tokenized, return_tensors="pt")
train_set = Dataset.from_dict(train_tokenized)
test_set = Dataset.from_dict(test_tokenized)
train_set = train_set.add_column("labels_cctop", train_cctops)
test_set = test_set.add_column("labels_cctop", test_cctops)
train_data_collator = DataCollatorForCCTOPModeling(tokenizer=tokenizer, mlm_probability=args.train_mlmp, pad_to_multiple_of=None)
test_data_collator = DataCollatorForCCTOPModeling(tokenizer=tokenizer, mlm_probability=args.test_mlmp, pad_to_multiple_of=None)

train_loader = DataLoader(train_set,  batch_size=args.batch_size, collate_fn=train_data_collator, shuffle=True)
test_loader = DataLoader(test_set,  batch_size=args.batch_size, collate_fn=test_data_collator, shuffle=True)

model_checkpoint = "facebook/esm2_t36_3B_UR50D"
model = AutoModelForMaskedLM.from_pretrained(model_checkpoint)
model.classifier = nn.Sequential(
    nn.Linear(model.config.hidden_size, model.config.hidden_size),
    nn.ReLU(),
    nn.Linear(model.config.hidden_size, num_labels),
    nn.Dropout(model.config.hidden_dropout_prob),
)
model = model.to(device)
# model.lm_head = EsmForMaskedLM(model.config)
params_1x, params_2x = [], []
for name, param in model.named_parameters():
    if name.startswith("esm.encoder.layer.35."):
        param.requires_grad = True
        params_1x.append(param)
    elif name.startswith("esm.encoder.emb_layer_norm_after"):
        param.requires_grad = True
        params_1x.append(param)
    elif name.startswith("esm.contact_head."):
        param.requires_grad = True
        params_1x.append(param)
    elif name.startswith("lm_head"):
        param.requires_grad = True
        params_2x.append(param)
    elif name.startswith("classifier"):
        param.requires_grad = True
        params_2x.append(param)
    else:
        param.requires_grad = False

wandb.init(mode='offline', project=args.project_name, name=args.run_name,
 config={"lr":args.lr,
        "epoch":args.epoch,
        "batch_size":args.batch_size,})
# wandb.watch(models=model, log='gradients', log_freq=args.watch_freq)

optimizer = optim.Adam([{'params': params_1x},
            {'params': params_2x, 'lr': args.lr * 2}], lr=args.lr)
token_acc,cctop_acc,token_loss_sum,cctop_loss_sum,perplexity = 0,0,0,0,0
logging.info("start training")
for epoch in range(args.epoch):
    model.train()
    for iteration, batch in enumerate(train_loader):
        if iteration % log_interval == 0 and iteration > 0:
            metric={
            # 'TRAIN/total_loss': total_loss_sum.item()/log_interval,
            'TRAIN/token_loss': token_loss_sum.item()/log_interval,
            "TRAIN/cctop_loss": cctop_loss_sum.item()/log_interval, 
            "TRAIN/loss": (token_loss_sum.item()+ ratio*cctop_loss_sum.item())/log_interval, 
            "TRAIN/perplexity": perplexity/log_interval,
            "TRAIN/token_accurracy": token_acc/log_interval,
            "TRAIN/cctop_accurracy": cctop_acc/log_interval,
            "epoch": epoch,
            }
            wandb.log(metric)
            token_acc,cctop_acc,token_loss_sum,cctop_loss_sum,perplexity = 0,0,0,0,0

        # outputs = model(**batch)
        # loss = outputs.loss
        for key in batch:
                batch[key] = batch[key].to(device)
        input_ids, attention_mask, token_label, cctop_label = batch['input_ids'], batch['attention_mask'], batch['labels'], batch['labels_cctop']
        outputs = model.esm(
                input_ids,
                attention_mask=attention_mask,
                return_dict=True,
            )
        sequence_output = outputs[0]
        token_logits = model.lm_head(sequence_output)
        cctop_logits = model.classifier(sequence_output)

        masked_lm_loss = None
        loss_fct = nn.CrossEntropyLoss()
        token_loss = loss_fct(token_logits.view(-1, num_tokens), token_label.view(-1))
        cctop_loss = loss_fct(cctop_logits.view(-1, num_labels), cctop_label.view(-1))
        total_loss = token_loss + ratio*cctop_loss
        token_loss_sum += token_loss
        cctop_loss_sum += cctop_loss
        perplexity += math.exp(token_loss)

        total_loss.backward()
        # 使用梯度累积
        if (iteration + 1) % args.accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
        logging.info(f'TRAIN: | epoch {epoch:3d} | {iteration:5d}/{len(train_loader):5d} batches | ')
            
        # scheduler.step()
        # compute accuracy
        token_acc += compute_accurracy(token_label, token_logits)
        cctop_acc += compute_accurracy(cctop_label, cctop_logits)

        # token_pred = torch.argmax(token_logits, dim=-1)
        # valid_token_indices = token_label == -100
        # masked_token_pred = token_pred[valid_token_indices]
        # masked_token_label = token_label[valid_token_indices]
        # token_correct += (masked_token_pred == masked_token_label).sum().item()
        # token_sum += masked_token_pred.numel()
        # valid_cctop_indices = cctop_label == -100
        # active_loss = attention_mask.view(-1) == 1
        # active_logits = prediction_cctop_scores.view(-1, num_labels)
        # active_labels = torch.where(
        # active_loss, labels_cctop.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels_cctop)
        #             )
        # loss = loss_fct(active_logits, active_labels)
    all_state_dict = model.state_dict()
    state_dict = {k:all_state_dict[k] for k in all_state_dict 
        if k.startswith("esm.encoder.layer.35.") or k.startswith("esm.encoder.emb_layer_norm_after")}
    if not os.path.exists(args.save_folder):
        os.makedirs(args.save_folder)
    torch.save({
                    'epoch': epoch,
                    'model_state_dict': state_dict,
                    'optimizer_state_dict': optimizer.state_dict()
                }, os.path.join(args.save_folder,f"epoch{epoch}.pt"))

    model.eval()
    with torch.no_grad():
        token_acc,cctop_acc,token_loss_sum,cctop_loss_sum,perplexity = 0,0,0,0,0
        for iteration, batch in enumerate(test_loader):
            for key in batch:
                batch[key] = batch[key].to(device)
        # outputs = model(**batch)
        # loss = outputs.loss
            input_ids, attention_mask, token_label, cctop_label = batch['input_ids'], batch['attention_mask'], batch['labels'], batch['labels_cctop']
            outputs = model.esm(
                    input_ids,
                    attention_mask=attention_mask,
                    return_dict=True,
                )
            sequence_output = outputs[0]
            token_logits = model.lm_head(sequence_output)
            cctop_logits = model.classifier(sequence_output)

            masked_lm_loss = None
            loss_fct = nn.CrossEntropyLoss()
            token_loss = loss_fct(token_logits.view(-1, num_tokens), token_label.view(-1))
            cctop_loss = loss_fct(cctop_logits.view(-1, num_labels), cctop_label.view(-1))
            total_loss = token_loss + ratio*cctop_loss
            token_loss_sum += token_loss
            cctop_loss_sum += cctop_loss
            perplexity += math.exp(token_loss)
            token_acc += compute_accurracy(token_label, token_logits)
            cctop_acc += compute_accurracy(cctop_label, cctop_logits)
        metric={
        # 'TEST/total_loss': total_loss.item()/len(test_loader),
        'TEST/token_loss': token_loss_sum.item()/(iteration+1),
        "TEST/cctop_loss": cctop_loss_sum.item()/(iteration+1), 
        "TEST/loss": (token_loss_sum.item() + ratio*cctop_loss_sum.item())/(iteration+1), 
        "TEST/perplexity": perplexity/(iteration+1),
        "TEST/token_accurracy": token_acc/(iteration+1),
        "TEST/cctop_accurracy": cctop_acc/(iteration+1),
        "epoch": epoch,
        }
        wandb.log(metric)
        token_acc,cctop_acc,token_loss_sum,cctop_loss_sum,perplexity = 0,0,0,0,0