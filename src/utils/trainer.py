import numpy as np
import torch
from tqdm import tqdm
import click

class Trainer:
  def __init__(self, model, optimizer, train_loader, val_loader, device, criterion, checkpoint_path):
    self.model = model.to(device)
    self.optimizer = optimizer
    self.train_loader = train_loader
    self.val_loader = val_loader
    self.device = device
    self.criterion = criterion
    self.checkpoint_path = checkpoint_path

  def _to_device(self, x):
    if torch.is_tensor(x):
      return x.to(self.device)
    if isinstance(x, dict):
      return {k: self._to_device(v) for k, v in x.items()}
    return x

  def _run_epoch(self, loader, train):
    self.model.train() if train else self.model.eval()
    ctx = torch.enable_grad() if train else torch.no_grad()

    total_loss = 0.0

    with ctx:
      for inputs in tqdm(loader):
        inputs = {k: self._to_device(v) for k, v in inputs.items()}
        prot = inputs["prot"]
        mol  = inputs["mol"]

        proj_prot, proj_mol = self.model(prot=prot, mol=mol)

        loss = self.criterion(proj_prot, proj_mol, inputs["positive_mask"])

        if train:
          self.optimizer.zero_grad()
          loss.backward()
          self.optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)

  def _train_epoch(self):
    return self._run_epoch(self.train_loader, train=True)

  def _eval_epoch(self):
    return self._run_epoch(self.val_loader, train=False)

  def bedroc(self, scores, labels, alpha=85):
    scores = np.asarray(scores)
    labels = np.asarray(labels)

    order = np.argsort(-scores)
    labels = labels[order]

    N = len(labels)
    n_actives = labels.sum()
    if n_actives == 0:
      return 0.0

    ranks = np.where(labels == 1)[0] + 1
    R_a   = n_actives / N

    exp_sum = np.sum(np.exp(-alpha * ranks / N))
    K1 = exp_sum / (R_a * (1 - np.exp(-alpha)) / (np.exp(alpha / N) - 1))
    K2 = (R_a * np.sinh(alpha / 2)) / (np.cosh(alpha / 2) - np.cosh(alpha / 2 - alpha * R_a))
    K3 = 1 / (1 - np.exp(alpha * (1 - R_a)))
    return K1 * K2 + K3


  def evaluate_(self, eval_batch_size=4):
    self.model.eval()
    dataset = self.val_loader.sampler.dataset
    prot_mode = self.model.protein_encoder.mode

    unique_mols = list(set(dataset.molecules_names))
    mol_embs = []
    with torch.no_grad():
      for i in range(0, len(unique_mols), eval_batch_size):
        batch = unique_mols[i:i + eval_batch_size]
        indices = [dataset.molecules_mapper[m] for m in batch]
        tensor = torch.tensor(dataset.X_mol[indices], device=self.device)
        mol_embs.append(self.model.encode_molecule(tensor))
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
            "input_ids": torch.stack(ids).to(self.device),
            "attention_mask": torch.stack(masks).to(self.device),
          }

        else:
          indices = [dataset.protein_to_idx[p] for p in batch]
          prot_input = torch.tensor(dataset.X_prot[indices], device=self.device)

        prot_embs.append(self.model.encode_protein(prot_input))
    z_prot = torch.cat(prot_embs)


    sim = z_prot @ z_mol.T

    val_prot_to_mols = {}
    for p, m in zip(dataset.proteins_names, dataset.molecules_names):
      val_prot_to_mols.setdefault(p, set()).add(m)

    mol_to_index = {m: i for i, m in enumerate(unique_mols)}
    val_prot_to_mol_idx = {
      p: {mol_to_index[m] for m in mols}
      for p, mols in val_prot_to_mols.items()
    }

    bedroc_total, count = 0.0, 0
    for i, prot in enumerate(unique_prots):
      scores = sim[i].cpu().numpy()
      labels = np.zeros(len(scores))
      for idx in val_prot_to_mol_idx[prot]:
        labels[idx] = 1
      bedroc_total += self.bedroc(scores, labels)
      count += 1

    return bedroc_total / count

  def train(self, num_epochs):
    best_metric = float("-inf")

    for epoch in range(num_epochs):
      click.echo(f"\nEpoch {epoch + 1}/{num_epochs}")

      train_loss = self._train_epoch()
      val_loss = self._eval_epoch()
      click.echo(f"Train loss: {train_loss:.4f}")
      click.echo(f"Val loss:   {val_loss:.4f}")

      if hasattr(self.criterion, "log_tau"):
        tau = torch.exp(self.criterion.log_tau).item()
        click.echo(f"Tau: {tau:.4f}")

      mean_bedroc = self.evaluate_()
      click.echo(f"BEDROC: {mean_bedroc:.4f}")

      if mean_bedroc > best_metric:
        best_metric = mean_bedroc
        torch.save(self.model.state_dict(), self.checkpoint_path)