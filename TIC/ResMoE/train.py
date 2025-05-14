import os
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

from TIC.utils.parameter import *
from TIC.utils.preprocess import get_dataset

logger = logging.getLogger(__name__)

def symmetric_cross_entropy(logits, targets, alpha=0.1, beta=1.0):
    ce = F.cross_entropy(logits, targets)
    reversed_logits = torch.flip(logits, dims=[1])
    reversed_targets = torch.flip(targets, dims=[1])
    reversed_ce = F.cross_entropy(reversed_logits, reversed_targets)
    return alpha * ce + beta * reversed_ce

def load_balance_loss(gate_weights, top_k_indeces, num_experts):
    expert_usages = torch.zeros(num_experts).to(gate_weights.device)
    for i in range(gate_weights.shape[0]):
        indices = top_k_indeces[i]
        expert_usages[indices] += gate_weights[i]
    expert_usage = F.softmax(expert_usages, dim=0)
    return -torch.sum(torch.log(expert_usage + 1e-8))

def total_loss(logits, targets, gate_weights, top_k_indeces, alpha=0.5):
    ce_loss = symmetric_cross_entropy(logits, targets)  
    balance_loss = load_balance_loss(gate_weights, top_k_indeces, logits.shape[1])
    return ce_loss + alpha * balance_loss

def get_checkpoint_path(epoch : int) -> str:
    return os.path.join(CHECKPOINT_DIR, f"ResMoE_epoch{epoch}.pth")

def dump_checkpoint(model : nn.Module, optimizer : nn.Module, epoch : int, loss : float):
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch" : epoch,
        "loss" : loss
    }
    torch.save(checkpoint, get_checkpoint_path(epoch))

def load_checkpoint(model : nn.Module, optimizer : nn.Module, epoch : int) -> float:
    '''
    Return:
        loss
    '''
    checkpoint = torch.load(get_checkpoint_path(epoch))
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    return checkpoint["loss"]

def train_epoch(model : nn.Module, optimizer : nn.Module, train_loader : torch.utils.data.DataLoader, epoch : int, logger : logging.Logger):
    loop = tqdm(train_loader, leave=True)
    for x, y in loop:
        with torch.no_grad():
        optimizer.zero_grad()
        logits = model(x)
        loss = total_loss(logits, y, model.gate_weights, model.top_k_indeces)
        loss.backward()
        optimizer.step()
        loop.set_description(f"Epoch {epoch}")
        loop.set_postfix(loss=loss.item())
