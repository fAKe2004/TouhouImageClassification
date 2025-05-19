import sys
from typing import Optional
from transformers import get_linear_schedule_with_warmup # Use AdamW and a suitable scheduler for Transformers

from TIC.utils.preprocess import get_dataset
from TIC.utils.parameter import *
from TIC.ViT.model import ViT # Import the ViT model

import torch
import torch.nn as nn
import torch.nn.functional as F

class SymmetricCrossEntropyLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=1.0, num_classes=120, reduction='mean', epsilon=1e-8):
        super(SymmetricCrossEntropyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.num_classes = num_classes
        self.reduction = reduction
        self.epsilon = epsilon
        # Standard CrossEntropyLoss for the CE part
        self.cross_entropy = nn.CrossEntropyLoss(reduction='none')

    def forward(self, pred, labels):

        ce = self.cross_entropy(pred, labels)

        pred = F.softmax(pred, dim=1)
        pred = torch.clamp(pred, min=self.epsilon, max=1.0)
        label_one_hot = F.one_hot(labels, self.num_classes).float().to(pred.device)
        label_one_hot = torch.clamp(label_one_hot, min=self.epsilon, max=1.0)
        rce = (-1 * torch.sum(pred * torch.log(label_one_hot), dim=1))

        loss = self.alpha * ce + self.beta * rce.mean()
        return loss.mean() if self.reduction == 'mean' else loss.sum()