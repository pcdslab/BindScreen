import torch
from torch.utils.data import Dataset
import numpy as np
from collections import defaultdict
import random


class MultiModalDataset(Dataset):
  def __init__(self,
               proteins_names,
               molecules_names,
               molecules_mapper,
               protein_mode,
               molecules_path='../embs/molecules.mmap',
               molecules_dim=768,
               proteins_sequences=None,
               proteins_tokenization=None,
               prot_max_len=1024,
               proteins_mapper=None,
               proteins_path='../embs/proteins.mmap',
               proteins_dim=2560):

    self.protein_mode = protein_mode

    self.proteins_names = proteins_names
    self.molecules_names = molecules_names
    self.molecules_mapper = molecules_mapper
    self.all_molecules = list(molecules_mapper.values())

    self.X_mol = np.memmap(molecules_path, dtype=np.float32, mode="r", shape=(len(molecules_mapper), molecules_dim))

    if self.protein_mode == "tokenized":
      unique_proteins = {}
      for name, seq in zip(proteins_names, proteins_sequences):
        if name not in unique_proteins:
          unique_proteins[name] = seq

      self.unique_protein_names = list(unique_proteins.keys())
      self.unique_protein_sequences = list(unique_proteins.values())
      
      tokens = proteins_tokenization(self.unique_protein_sequences, padding="max_length", truncation=True, max_length=prot_max_len, return_tensors="pt")
      self.prot_input_ids = tokens["input_ids"]
      self.prot_attention_mask = tokens["attention_mask"]
      
      self.protein_to_idx = {p: i for i, p in enumerate(self.unique_protein_names)}

    else:
      self.X_prot = np.memmap(proteins_path, dtype=np.float32, mode="r", shape=(len(proteins_mapper), proteins_dim))
      self.protein_to_idx = proteins_mapper

    self.protein_to_mols = defaultdict(list)
    for p, m in zip(proteins_names, molecules_names):
      p_idx = self.protein_to_idx[p]
      m_idx = molecules_mapper[m]
      self.protein_to_mols[p_idx].append(m_idx)


  def get_molecule(self, mol_idx):
    return torch.from_numpy(np.array(self.X_mol[mol_idx], dtype=np.float32, copy=True))

  def get_protein(self, prot_idx):
    if self.protein_mode == "tokenized":
      return {
        "input_ids": self.prot_input_ids[prot_idx],
        "attention_mask": self.prot_attention_mask[prot_idx],
      }
    else:
      return torch.from_numpy(np.array(self.X_prot[prot_idx], dtype=np.float32, copy=True))

  def __len__(self):
    return len(self.protein_to_idx)


class ChunkBatchSampler:
  def __init__(self, dataset, chunk_size=32, batch_size=2048, chunks_per_batch=32):
    self.dataset = dataset
    self.chunk_size = chunk_size
    self.batch_size = batch_size
    self.chunks_per_batch = chunks_per_batch

    self.chunks = []
    for p, ligands in dataset.protein_to_mols.items():
      for i in range(0, len(ligands), chunk_size):
        self.chunks.append((p, ligands[i:i + chunk_size]))

  def shuffle(self):
    random.shuffle(self.chunks)

  def __len__(self):
    return len(self.chunks) // self.chunks_per_batch

  def _build_molecule_batch(self, mol_list):
    seen = set()
    mol_unique = []

    for m in mol_list:
      if m not in seen:
        mol_unique.append(m)
        seen.add(m)
    
    while len(mol_unique) < self.batch_size:
      m = random.choice(self.dataset.all_molecules)
      if m not in seen:
        mol_unique.append(m)
        seen.add(m)
    
    return mol_unique[: self.batch_size]

  def get_batch(self, start):
    selected_chunks = self.chunks[start:start + self.chunks_per_batch]

    proteins, mol_list = [], []
    for p, chunk in selected_chunks:
      proteins.append(p)
      mol_list.extend(chunk)

    prot_unique = list(set(proteins))
    mol_unique = self._build_molecule_batch(mol_list)

    if self.dataset.protein_mode == "tokenized":
      prot_ids, prot_masks = [], []
      for p in prot_unique:
        prot = self.dataset.get_protein(p)
        prot_ids.append(prot["input_ids"])
        prot_masks.append(prot["attention_mask"])

      prot_batch = {
        "input_ids": torch.stack(prot_ids),
        "attention_mask": torch.stack(prot_masks),
      }

    else:
      prot_batch = torch.stack([self.dataset.get_protein(p) for p in prot_unique])

    mol_batch = torch.stack([self.dataset.get_molecule(m) for m in mol_unique])

    positive_mask = torch.zeros(len(prot_unique), len(mol_unique))
    for i, p in enumerate(prot_unique):
      positives = set(self.dataset.protein_to_mols[p])
      for j, m in enumerate(mol_unique):
        if m in positives:
          positive_mask[i, j] = 1

    return {
      "prot": prot_batch,
      "mol": mol_batch,
      "positive_mask": positive_mask,
    }


class ProteinBatchLoader:
  def __init__(self, sampler):
    self.sampler = sampler

  def __iter__(self):
    self.sampler.shuffle()
    for i in range(0, len(self.sampler.chunks), self.sampler.chunks_per_batch):
      yield self.sampler.get_batch(i)

  def __len__(self):
    return len(self.sampler)