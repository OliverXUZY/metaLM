
import torch
import torch.nn as nn
import torch.nn.functional as F


class FSCentroidClassifier(nn.Module):
  def __init__(self, temp=1., learn_temp=False):
    super(FSCentroidClassifier, self).__init__()

    self.temp = temp
    if learn_temp:
      self.temp = nn.Parameter(torch.tensor(temp))

  def forward(self, s, q):
    #############################################
    # Y: number of ways / categories per task
    # S: number of shots per category
    # Q: number of queries per category
    # D: input / feature dimension
    #############################################
    assert s.dim() == 3                             # [Y, S, D]
    assert q.dim() == 2                             # [Y * Q, D]
    
    centroid = torch.mean(s, dim=1)                 # [Y, D]
    centroid = F.normalize(centroid, dim=-1)        # [Y, D]
    centroid = centroid.transpose(-2, -1)           # [D, Y]
    q = F.normalize(q, dim=-1)                      # [Y * Q, D]
    logits = torch.matmul(q, centroid)              # [Y * Q, Y]
    logits = logits * self.temp
    return logits
