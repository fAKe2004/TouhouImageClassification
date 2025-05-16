import torch
import torch.nn as nn
import torch.nn.functional as F

import TIC.ViT.model as ViT

class TreeModule(nn.Module):

    def __init__(self, root : nn.Module, sons : nn.ModuleList, top_k : int):
        super(TreeModule, self).__init__()
        self.root = root
        self.sons = sons
        self.top_k = top_k

    def forward(self, x):
        choose = self.root(x)   # (B, S)
        top_k_weights, top_k_indeces = torch.topk(choose, k = self.top_k, dim = 1)  # (B, K), (B, K)
        smoothed = F.softmax(top_k_weights, dim = 1)    # (B, K)
        son_logits = torch.cat([
            torch.stack([self.sons[i.item()](x[b:b+1]) for i in top_k_indeces[b]], dim = 1) # (1, K, C)
            for b in range(x.shape[0])
        ], dim = 0)
        return torch.bmm(smoothed.unsqueeze(1), son_logits)

def make_TreeViT(num_categories : int, num_classes : int, top_k : int, root_pretrained : bool, sons_pretrained : bool):
    return TreeModule(
        root = ViT.ViT(num_classes = num_categories, pretrained = root_pretrained),
        sons = nn.ModuleList([ViT.ViT(num_classes = num_classes, pretrained = sons_pretrained) for _ in range(num_categories)]),
        top_k = top_k
    )
