import torch
from transformers import AutoModel, AutoTokenizer
import pandas as pd
import numpy as np
from tqdm import tqdm
import json
from datasets import load_dataset
import click
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
os.makedirs('../embs', exist_ok=True)

@click.command()
@click.option("--mode", type=click.Choice(["protein", "molecule"]), required=True)
def main(mode):

  dataset_chembl = load_dataset('SaeedLab/SeqScreen', data_dir='chembl')
  dataset_lit = load_dataset('SaeedLab/SeqScreen', data_dir='lit_pcba')

  full_data = pd.concat([dataset_chembl['train'].to_pandas(),
                         dataset_chembl['validation'].to_pandas(),
                         dataset_chembl['test'].to_pandas(),
                         dataset_lit['train'].to_pandas(),
                         dataset_lit['validation'].to_pandas(),
                         dataset_lit['test'].to_pandas()])

  if mode == 'protein':
    MAX_LEN = 1022
    EMB_DIM = 2560
    mapper_path = '../embs/proteins_mapping.json'
    data_path = '../embs/proteins.mmap'

    tokenizer = AutoTokenizer.from_pretrained('facebook/esm2_t36_3B_UR50D')
    model = AutoModel.from_pretrained("facebook/esm2_t36_3B_UR50D").to(device).eval()

    df = full_data.groupby('prot_id')['prot'].first().sort_index().reset_index()
    mapper = {str(name): i for i, name in enumerate(df.prot_id.values)}
    with open(mapper_path, 'w') as f:
      json.dump(mapper, f)

    X = np.memmap(data_path, dtype=np.float32, mode='w+', shape=(len(df), EMB_DIM))
    
    with torch.inference_mode():
      for name, seq in tqdm(df.values, total=len(df)):
        idx = mapper[str(name)]
        protein = " ".join(seq[:MAX_LEN])
        ids = tokenizer.batch_encode_plus([protein], add_special_tokens=True, return_tensors='pt').to(device)
        embedding_rpr = model(**ids)
        embs = embedding_rpr.last_hidden_state[0, :].detach().cpu().numpy().mean(axis=0)
        X[idx] = embs
    X.flush()

  else:
    EMB_DIM = 768
    mapper_path = '../embs/molecules_mapping.json'
    data_path = '../embs/molecules.mmap'

    tokenizer = AutoTokenizer.from_pretrained("SaeedLab/MolDeBERTa-base-123M-mlc")
    model = AutoModel.from_pretrained("SaeedLab/MolDeBERTa-base-123M-mlc").to(device).eval()

    df = full_data.groupby('mol_id')['mol'].first().sort_index().reset_index()
    mapper = {str(name): i for i, name in enumerate(df.mol_id.values)}
    with open(mapper_path, 'w') as f:
      json.dump(mapper, f)

    X = np.memmap(data_path, dtype=np.float32, mode='w+', shape=(len(df), EMB_DIM))
    
    with torch.inference_mode():
      for name, seq in tqdm(df.values, total=len(df)):
        idx = mapper[str(name)]
        ids = tokenizer([seq], return_tensors='pt', truncation=True, max_length=128).to(device)
        embedding_rpr = model(**ids)
        embs = embedding_rpr.last_hidden_state[0, :].detach().cpu().numpy().mean(axis=0)
        X[idx] = embs
    X.flush()


if __name__ == '__main__':
  main()