import math
import json
import torch
import wandb
import argparse
# %env WANDB_PROJECT=ft_plm_1
parser = argparse.ArgumentParser()
import numpy as np
parser = argparse.ArgumentParser()
parser.add_argument('--epoch',type=int,default=10,help="epochs to train the model")
parser.add_argument('--batch_size',type=int,default=8,help="batch size tokens")
parser.add_argument('--weight_decay',type=float,default=0.01, help="weight_decay")
parser.add_argument('--jsonl',type=str,default="/lustre/gst/xuchunfu/zhangxt/data/tmpnn_v8.jsonl", help="weight_decay")
parser.add_argument('--project_name',type=str,default="aaa", help="name")
parser.add_argument('--run_name',type=str,default="aaa", help="name")
parser.add_argument('--watch_freq',type=int,default=500, help="watch")
parser.add_argument('--lr',type=float,default=5e-4, help="learning rate of Adam optimizer")
parser.add_argument('--log_interval',type=int,default=2, help="learning rate of Adam optimizer")
parser.add_argument('--save_folder',type=str,default="/lustre/gst/xuchunfu/zhangxt/plm/0/", help="learning rate of Adam optimizer")
parser.add_argument('--ratio',type=float,default=0.2, help="learning rate of Adam optimizer")
parser.add_argument('--mlmp',type=float,default=0.15, help="learning rate of Adam optimizer")
# parser.add_argument('--write_path',type=str,default="/lustre/gst/xuchunfu/zhangxt/TMPNN/soluble/15B_plddt_design/", help="name")
# parser.add_argument('--read_path',type=str,default="/lustre/gst/xuchunfu/zhangxt/TMPNN/soluble/original_design/", help="learning rate of Adam optimizer")
# parser.add_argument('--pdb_path',type=str,default="/lustre/gst/xuchunfu/zhangxt/TMPNN/soluble/predict/original_design/", help="learning rate of Adam optimizer")
parser.add_argument('--model',type=str,default="mlmp30", help="learning rate of Adam optimizer")
parser.add_argument('--parameters',type=str,default=None, help="learning rate of Adam optimizer")

args = parser.parse_args()
def load_jsonl(json_file) -> list:
    seq, cctop = [], []
    cctop_code = 'IMOULS'

    with open(json_file, "r") as f:
        for line in f:

                # 每一行代表一个序列的字典字符串加一个'\n',对于字符串应该用json.loads()文件进行读取
                data = json.loads(line.replace("\n", ""))
                seq.append(data["seq"])

                cctop_i = data["cctop"]
                cctop_i.replace('I','A')
                cctop_i.replace('M','B')
                cctop_i.replace('O','C')
                cctop_i.replace('U','D')
                cctop_i.replace('L','E')
                cctop_i.replace('S','F')
                cctop.append(cctop_i)
                # cctop.append([cctop_code.index(i) for i in data["cctop"]])
           
    return seq, cctop

def group_texts(examples):
    # Concatenate all texts.
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
    # customize this part to your needs.
    if total_length >= block_size:
        total_length = (total_length // block_size) * block_size
    # Split by chunks of block_size.
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result

def generate_labels(examples):
    result = {k: examples[k] for k in examples.keys()}
    result["labels"] = result["input_ids"].copy()
    return result

seq, cctop = load_jsonl(args.jsonl)

from sklearn.model_selection import train_test_split
train_sequences, test_sequences, train_cctops, test_cctops = train_test_split(seq, cctop, test_size=0.3, shuffle=True)
from transformers import AutoTokenizer
model_checkpoint = args.parameters

# 不采用tokenizer.tokenie进行批处理，因此这里不需要padding or truncation，每个句子是一个数字列表
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
train_tokenized = tokenizer(train_sequences)
test_tokenized = tokenizer(test_sequences)
from datasets import Dataset
train_dataset = Dataset.from_dict(train_tokenized)
test_dataset = Dataset.from_dict(test_tokenized)


# block_size = 38
# train_dataset = train_dataset.map(group_texts, batched=True, num_proc=4)
# train_dataset = train_dataset.map(generate_labels, batched=True, num_proc=4)
# test_dataset = test_dataset.map(group_texts, batched=True, num_proc=4)
from transformers import DataCollatorForLanguageModeling,DataCollatorForTokenClassification
# tokenizer.pad_token = tokenizer.eos_token
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=args.mlmp)
# data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

from transformers import AutoModelForTokenClassification,AutoModelWithLMHead,AutoModelForMaskedLM, TrainingArguments, Trainer
model = AutoModelForMaskedLM.from_pretrained(model_checkpoint)
# # Add 1 since 0 can be a label
for name, param in model.named_parameters():
    if name.startswith("esm.encoder.layer.35."):
        param.requires_grad = True
    elif name.startswith("esm.encoder.emb_layer_norm_after"):
        param.requires_grad = True
    elif name.startswith("esm.contact_head."):
        param.requires_grad = True
    elif name.startswith("lm_head"):
        param.requires_grad = True
    else:
        param.requires_grad = False
device=torch.device('cuda')
model=model.to(device)

wandb.init(mode='offline', project=args.project_name, name=args.run_name, config={"lr":args.lr,
        "epoch":args.epoch,
        "batch_size":args.batch_size,})



def compute_metrics(p):
    logits, label = p
    logits, label = torch.from_numpy(logits), torch.from_numpy(label)
    pred = torch.argmax(logits, dim=-1)
    valid_indices = label != -100
    masked_pred = pred[valid_indices]
    masked_label = label[valid_indices]
    correct = (masked_pred == masked_label).sum().item()
    num = masked_pred.numel()
    return {'accuracy': correct/num}
train_args = TrainingArguments(
    output_dir=args.save_folder,
    # output_dir="/pubhome/xtzhang/result/save",
    evaluation_strategy = "epoch",
    save_strategy = "epoch",
    learning_rate=args.lr,
    per_device_train_batch_size=args.batch_size,
    per_device_eval_batch_size=args.batch_size,
    num_train_epochs=args.epoch,
    weight_decay=args.weight_decay,
    load_best_model_at_end=True,
    run_name=args.run_name,
    # gradient_accumulation_steps = 2,
    # metric_for_best_model="accuracy",
    # gradient_checkpointing=True,
    # push_to_hub=True,
)
trainer = Trainer(
    model,
    train_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)
with torch.no_grad():
    eval_results = trainer.evaluate()
print(args.model)
print(f"Perplexity: {math.exp(eval_results['eval_loss']):.2f}")
