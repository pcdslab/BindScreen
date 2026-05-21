import torch
import torch.nn as nn
import torch.nn.functional as F


class ProjectionLayer(nn.Module):
  def __init__(self, in_dim, out_dim, dropout=0.1):
    super().__init__()
    self.projection = nn.Sequential(
      nn.Linear(in_dim, out_dim),
      nn.LayerNorm(out_dim),
      nn.GELU(),
      nn.Dropout(dropout),
      nn.Linear(out_dim, out_dim),
    )

  def forward(self, x):
    return F.normalize(self.projection(x), dim=-1)


class ProteinEncoder(nn.Module):
  def __init__(self, mode, encoder_model=None):
    super().__init__()
    self.mode = mode
    self.encoder = encoder_model

  def _mean_pooling(self, hidden, attention_mask):
    mask = attention_mask.unsqueeze(-1).float()
    return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-8)

  def forward(self, embedding=None, input_ids=None, attention_mask=None):
    if self.mode == "embedding":
      return embedding
    else:
      hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
      return self._mean_pooling(hidden, attention_mask)

class MultiModalModel(nn.Module):
  def __init__(self, protein_encoder, prot_dim=2560, mol_dim=768, proj_dim=512, dropout=0.1):
    super().__init__()
    self.protein_encoder = protein_encoder
    self.proj_prot = ProjectionLayer(prot_dim, proj_dim, dropout)
    self.proj_mol = ProjectionLayer(mol_dim, proj_dim, dropout)

  def encode_protein(self, prot):
    if isinstance(prot, dict):
      rep = self.protein_encoder(input_ids=prot["input_ids"], attention_mask=prot["attention_mask"])
    else:
      rep = self.protein_encoder(embedding=prot)
    return self.proj_prot(rep)

  def encode_molecule(self, mol):
    return self.proj_mol(mol)

  def forward(self, prot, mol):
    return self.encode_protein(prot), self.encode_molecule(mol)