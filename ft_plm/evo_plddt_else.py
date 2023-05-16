import os
import torch
# os.environ["CUDA_VISIBLE_DEVICES"]="3"
from transformers import AutoTokenizer,AutoModelForMaskedLM
import argparse
import logging
from biotite.structure.residues import get_residues
from biotite.structure import get_chains
from biotite.sequence import ProteinSequence
import biotite.structure as struc
import biotite.structure.io as strucio
import numpy as np
parser = argparse.ArgumentParser()
parser.add_argument('--write_path',type=str,default="/lustre/gst/xuchunfu/zhangxt/TMPNN/soluble/15B_plddt_design/", help="name")
parser.add_argument('--read_path',type=str,default="/lustre/gst/xuchunfu/zhangxt/TMPNN/soluble/original_design/", help="learning rate of Adam optimizer")
parser.add_argument('--pdb_path',type=str,default="/lustre/gst/xuchunfu/zhangxt/TMPNN/soluble/predict/original_design/", help="learning rate of Adam optimizer")
parser.add_argument('--model',type=str,default="/lustre/gst/xuchunfu/zhangxt/.cache/huggingface/hub/models--facebook--esm2_t48_15B_UR50D/snapshots/5fbca39631164edc1d402a5aa369f982f72ee282/", help="learning rate of Adam optimizer")

args = parser.parse_args()

def extract_plddt(protein,chain_id=None):

    if isinstance(protein,str):
        # model = 1 to load a AtomArray object
        # extra_fields to load the b_factor column
        atom_array = strucio.load_structure(protein,model=1,extra_fields=["b_factor"])
    elif isinstance(protein, struc.AtomArrayStack):
        atom_array = protein[0]
    elif isinstance(protein, struc.AtomArray):
        atom_array = protein

    # add multiple chain sequence subtract function
    all_chains = get_chains(atom_array)
    if len(all_chains) == 0:
        raise ValueError('No chains found in the input file.')
    if chain_id is None:
        chain_ids = all_chains
    elif isinstance(chain_id, list):
        chain_ids = chain_id
    else:
        chain_ids = [chain_id] 
    for chain in chain_ids:
        if chain not in all_chains:
            raise ValueError(f'Chain {chain} not found in input file')
    chain_filter = [a.chain_id in chain_ids for a in atom_array]
    atom_array = atom_array[chain_filter]
    # mask canonical aa 
    aa_mask = struc.filter_amino_acids(atom_array)
    atom_array = atom_array[aa_mask]
    # ca atom only
    atom_array = atom_array[atom_array.atom_name == "CA"]
    plddt = np.array([i.b_factor for i in atom_array])
    return plddt, np.mean(plddt)
for name in ('6DKM_A', '6DKM_B', '6DLM_A', '6DLM_B'):
    write_path = args.write_path + name + '.fa'
    # read_path = args.read_path
    read_path = args.read_path + name + '.fa'
    # name = read_path[-7:-3]
    pdb_dir = args.pdb_path + name + '/pdbs/'
    pdb_path_list = os.listdir(pdb_dir)
    pdb_path_list.sort(key=lambda x:int(x.split('.')[0])) #对‘.’进行切片，并取列表的第一个值（左边的文件名）转化整数型
    plddt_list = []
    for pdb in pdb_path_list:
        pdb_path = pdb_dir + pdb
        plddt, mean_plddt = extract_plddt(pdb_path)
        plddt_list.append(list(plddt))
    # plddts = torch.tensor(plddt_list).cuda()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForMaskedLM.from_pretrained(args.model)
    model=model.cuda()
    model=model.eval()
    softmax=torch.nn.Softmax(dim=-1)
    with open(read_path, 'r') as f:
        seqs = f.readlines()
    update_p = 0.1
    print('starting predict')
    

    seq_packed = [(seqs[i], seqs[i+1]) for i in range(0, len(seqs),2)]



    with open(write_path, 'a') as f:
        i = -1
        for seq_2, plddt in zip(seq_packed, plddt_list):
            i += 2
            plddt = torch.tensor(plddt).cuda()
            for seq in seq_2:
                if not seq.startswith('>'):
                    with torch.no_grad():
                        seq = seq[:-1] # 去掉换行符
                        inputs = tokenizer(seq, return_tensors="pt")
                        for key in inputs:
                                inputs[key] = inputs[key].cuda()
                        logits = model(**inputs).logits[0, ...]
                        p = softmax(logits)[1:-1]
                        # indices[L]
                        indices=inputs["input_ids"].squeeze(0)[1:-1]
                        # selected_elements[L,]
                        selected_elements = torch.gather(p, dim=1, index=indices.unsqueeze(1)).squeeze(1)
                        # plddt???
                        replace_indices = (selected_elements < update_p) & (plddt < 50)
                        # replace_indices = plddt < 50
                        max_indices = torch.argmax(p, dim=1)
                        indices[replace_indices] = max_indices[replace_indices]
                        new_selected_elements = torch.gather(p, dim=1, index=indices[:,None])
                        p_mean = torch.mean(new_selected_elements)
                        seqs[i] = ''.join(tokenizer.convert_ids_to_tokens(indices)) + '\n'
                        title = seqs[i-1][:-1] 
                        seqs[i-1] = title[:title.rfind(',')] + ',' + str(round(float(p_mean), 4)) + '\n'

        f.truncate(0)
        for seq in seqs:       
            f.write(seq)
        print('finish')
