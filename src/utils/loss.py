import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiPositiveInfoNCE(nn.Module):
  def __init__(self, init_temperature=0.1):
    super().__init__()
    self.log_tau = nn.Parameter(torch.log(torch.tensor(init_temperature)))

  def forward(self, z, y, positive_mask):
    tau = torch.exp(self.log_tau)

    positive_mask = positive_mask / (positive_mask.sum(dim=1, keepdim=True) + 1e-8)

    logits = torch.matmul(z, y.T)
    logits = logits / tau
    
    loss = -(positive_mask * F.log_softmax(logits, dim=1)).sum(dim=1)

    return loss.mean()