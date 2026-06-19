import pandas as pd
import numpy as np
import torch
import os
import warnings
import json
import click
from collections import defaultdict
from sklearn.metrics import roc_auc_score
from transformers import AutoModel, AutoTokenizer
from peft import LoraConfig, get_peft_model
from datasets import load_dataset 
import os

from utils.model import *
from utils.trainer import *
from utils.dataset import *
from utils.utils import *
from utils.loss import *

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
warnings.filterwarnings("ignore", category=FutureWarning, module="torch")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
os.makedirs('../models', exist_ok=True)


def bedroc(scores, labels, alpha=85):
  scores = np.asarray(scores)
  labels = np.asarray(labels)

  order = np.argsort(-scores)
  labels = labels[order]

  N, n_actives = len(labels), labels.sum()
  if n_actives == 0:
    return 0.0

  ranks = np.where(labels == 1)[0] + 1
  R_a = n_actives / N
  exp_sum = np.sum(np.exp(-alpha * ranks / N))

  K1 = exp_sum / (R_a * (1 - np.exp(-alpha)) / (np.exp(alpha / N) - 1))
  K2 = (R_a * np.sinh(alpha / 2)) / (np.cosh(alpha / 2) - np.cosh(alpha / 2 - alpha * R_a))
  K3 = 1 / (1 - np.exp(alpha * (1 - R_a)))
  return K1 * K2 + K3


def evaluate(model, 
             loader, 
             device, 
             evaluation_dataset, 
             df, 
             eval_batch_size=4):

  model.eval()
  dataset = loader.sampler.dataset
  prot_mode = model.protein_encoder.mode

  unique_mols = list(set(dataset.molecules_names))

  mol_embs = []
  with torch.no_grad():
    for i in range(0, len(unique_mols), eval_batch_size):
      batch = unique_mols[i:i + eval_batch_size]
      indices = [dataset.molecules_mapper[m] for m in batch]
      tensor  = torch.tensor(dataset.X_mol[indices], device=device)
      mol_embs.append(model.encode_molecule(tensor))
  z_mol = torch.cat(mol_embs)

  if prot_mode == "tokenized":
    unique_prots = dataset.unique_protein_names
  else:
    unique_prots = list(set(dataset.proteins_names))
  
  prot_embs = []
  with torch.no_grad():
    for i in range(0, len(unique_prots), eval_batch_size):
      batch = unique_prots[i:i + eval_batch_size]
      
      if prot_mode == "tokenized":
        ids, masks = [], []
        for p in batch:
          tok = dataset.get_protein(dataset.protein_to_idx[p])
          ids.append(tok["input_ids"])
          masks.append(tok["attention_mask"])
        prot_input = {
          "input_ids":    torch.stack(ids).to(device),
          "attention_mask": torch.stack(masks).to(device),
        }

      else:
        indices  = [dataset.protein_to_idx[p] for p in batch]
        prot_input = torch.tensor(dataset.X_prot[indices], device=device)
      
      prot_embs.append(model.encode_protein(prot_input))
  
  z_prot = torch.cat(prot_embs)
  sim = z_prot @ z_mol.T

  if evaluation_dataset == 'lit_pcba':
    prot_to_row = {p: i for i, p in enumerate(unique_prots)}
    mol_to_col = {str(m): j for j, m in enumerate(unique_mols)}
    grouped = df.groupby("prot_id")
    metrics_sum = defaultdict(float)
    count = 0

    for prot in unique_prots:
      test_df_prot = grouped.get_group(prot)
      mol_ids = test_df_prot.mol_id.values
      labels  = test_df_prot.active.values.astype(float)

      row = prot_to_row[prot]
      col_indices = [mol_to_col[str(mid)] for mid in mol_ids]

      scores = sim[row][col_indices].cpu().numpy()
      n_actives = labels.sum()

      order = np.argsort(-scores)
      ranked_labels = labels[order]

      for frac in [0.005, 0.01, 0.05]:
        k = max(1, int(frac * len(labels)))
        hits = ranked_labels[:k].sum()
        ef = (hits / k) / (n_actives / len(labels))
        metrics_sum[f"EF{frac*100}%"] += ef

      metrics_sum["AUROC"]  += roc_auc_score(labels, scores)
      metrics_sum["BEDROC85"] += bedroc(scores, labels, 85)
      count += 1

    metrics = {k: v / count for k, v in metrics_sum.items()}

  else:
    metrics_sum = defaultdict(float)
    n_queries = len(unique_prots)

    val_prot_to_mols = {}
    for p, m in zip(dataset.proteins_names, dataset.molecules_names):
      val_prot_to_mols.setdefault(p, set()).add(m)

    mol_to_index = {m: i for i, m in enumerate(unique_mols)}
    val_prot_to_mol_idx = {
      p: {mol_to_index[m] for m in mols}
      for p, mols in val_prot_to_mols.items()
    }

    for i, prot in enumerate(unique_prots):
      scores = sim[i].cpu().numpy()
      labels = np.zeros(len(scores))
      for idx in val_prot_to_mol_idx[prot]:
        labels[idx] = 1

      order = np.argsort(-scores)
      ranked_labels = labels[order]

      n_actives = labels.sum()

      for frac in [0.005, 0.01, 0.05]:
        k = max(1, int(frac * len(labels)))
        hits = ranked_labels[:k].sum()
        ef = (hits / k) / (n_actives / len(labels))

        metrics_sum[f"EF{frac*100}%"] += ef

      metrics_sum["AUROC"] += roc_auc_score(labels, scores)
      metrics_sum["BEDROC85"] += bedroc(scores, labels, 85)

    metrics = {k: v / n_queries for k,v in metrics_sum.items()}

  return metrics

def make_dataset(df, common, extra):
  return MultiModalDataset(
    proteins_names=df.prot_id.values,
    molecules_names=df.mol_id.values,
    proteins_sequences=df.prot.values,
    **common,
    **extra)

@click.command()
@click.option("--mode", type=click.Choice(["embedding", "tokenized"]), required=True)
@click.option("--dataset", type=click.Choice(["chembl", "lit_pcba"]), required=True)
def main(mode, dataset):
  set_seed(42)

  dataset_dict = load_dataset('SaeedLab/BindScreen', data_dir=dataset)

  train_df = dataset_dict['train'].to_pandas()
  val_df = dataset_dict['validation'].to_pandas()
  test_df = dataset_dict['test'].to_pandas()

  with open('../embs/molecules_mapping.json') as f:
    molecule_mapping = json.load(f)

  with open('../embs/proteins_mapping.json') as f:
    protein_mapping = json.load(f)

  common = dict(
    molecules_mapper=molecule_mapping,
    protein_mode=mode,
    proteins_mapper=protein_mapping
  )

  if mode == "embedding":
    extra = dict()

  else:
    tokenizer = AutoTokenizer.from_pretrained('facebook/esm2_t36_3B_UR50D')
    encoder_model = AutoModel.from_pretrained('facebook/esm2_t36_3B_UR50D', torch_dtype=torch.bfloat16)

    lora_cfg = LoraConfig(
      r=16, lora_alpha=32,
      target_modules=["query", "key", "value"],
      lora_dropout=0.05, bias="none",
      task_type="FEATURE_EXTRACTION",
    )
    encoder_model = get_peft_model(encoder_model, lora_cfg)
    encoder_model.enable_input_require_grads()
    encoder_model.gradient_checkpointing_enable()
    encoder_model.print_trainable_parameters()

    encoder_model = encoder_model.to(device)
    extra = dict(proteins_tokenization=tokenizer)

  train_dataset = make_dataset(train_df, common, extra)
  val_dataset = make_dataset(val_df, common, extra)
  test_dataset = make_dataset(test_df, common, extra)

  train_loader = ProteinBatchLoader(ChunkBatchSampler(train_dataset))
  val_loader = ProteinBatchLoader(ChunkBatchSampler(val_dataset))
  test_loader = ProteinBatchLoader(ChunkBatchSampler(test_dataset))

  if mode == "tokenized":
    protein_encoder = ProteinEncoder(mode="tokenized", encoder_model=encoder_model)
  else:
    protein_encoder = ProteinEncoder(mode="embedding")

  model = MultiModalModel(protein_encoder=protein_encoder).to(device)

  criterion = MultiPositiveInfoNCE()
  params = [
    {"params": model.proj_prot.parameters(),  "lr": 1e-4},
    {"params": model.proj_mol.parameters(),   "lr": 1e-4},
    {"params": criterion.parameters(),    "lr": 1e-4},
  ]
  if mode == "tokenized":
    params.append({"params": encoder_model.parameters(), "lr": 1e-5})

  optimizer = torch.optim.AdamW(params)

  checkpoint_path = f'../models/{mode}-{dataset}.pt'

  trainer = Trainer(
    model=model, 
    optimizer=optimizer,
    train_loader=train_loader, 
    val_loader=val_loader,
    device=device, 
    criterion=criterion,
    checkpoint_path=checkpoint_path,
  )
  trainer.train(num_epochs=30)

  model.load_state_dict(torch.load(checkpoint_path, map_location=device))

  metrics = evaluate(model=model, 
                     loader=test_loader, 
                     device=device, 
                     evaluation_dataset=dataset,
                     df=test_df)

  click.echo("\n--- Metrics ---")
  for name, value in metrics.items():
    click.echo(f"{name}: {value:.4f}")

if __name__ == "__main__":
  main()