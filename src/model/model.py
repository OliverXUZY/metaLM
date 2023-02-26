import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as ds
import torch.autograd as autograd

class Model(nn.Module):
    def __init__(self, enc, head, n_shot = 2):
        super(Model, self).__init__()
        self.enc = enc
        self.head = head
        self.n_shot = n_shot
    
    def forward(self, sq_idx, label_idx, batch):
        # make sure idx vector are 1-dim
        assert sq_idx.dim() == 1
        assert label_idx.dim() == 1
        out = self.enc(**batch)               # [30, 768] [Y * (S + Q), D]
        num_classes = label_idx.unique().shape[0]
        # print("num_classes: ", num_classes)
        query = out[sq_idx == 1,:] # torch.Size([24, 768])  [Y * Q, D]
        shot = out[sq_idx == 0,:].view(num_classes,self.n_shot,-1)  # torch.Size([3, 2, 768]) [Y, S, D]
        # print(shot.shape, query.shape)
        logits = self.head(shot, query)   # [Y * Q, Y]

        return logits
