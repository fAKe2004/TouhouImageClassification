import os
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import lightning as L

from TIC.ResMoE.model import MoEClassifier, make_ViTMoE
from TIC.utils.parameter import *
from TIC.utils.preprocess import get_dataset
from .parameter import *

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
        expert_usages += gate_weights[i]
    expert_usage = F.softmax(expert_usages, dim=0)
    return -torch.sum(torch.log(expert_usage + 1e-8))

def total_loss(logits, targets, gate_weights, top_k_indeces, alpha=0.5):
    ce_loss = symmetric_cross_entropy(logits, targets)  
    balance_loss = load_balance_loss(gate_weights, top_k_indeces, gate_weights.shape[1])
    return ce_loss + alpha * balance_loss

class ResMoETrainerModule(L.LightningModule):

    def __init__(self, model : MoEClassifier, optimizer : optim.Optimizer):
        super().__init__()
        self.model = model
        self.optimizer = optimizer

    def training_step(self, batch, batch_idx):
        x, y = batch
        targets = F.one_hot(y, num_classes = self.model.num_classes).float()
        logits, gate_weights, top_k_indeces = self.model(x)
        loss = total_loss(logits, targets, gate_weights, top_k_indeces)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        return self.optimizer

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

def train_epoch(model : nn.Module, optimizer : optim.Optimizer, train_loader : torch.utils.data.DataLoader, epoch : int, logger : logging.Logger):
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

if __name__ == '__main__':
    model = make_ViTMoE(
        num_classes = NUM_CLASSES,
        num_experts = MOE_NUM_EXPERTS,
        top_k = MOE_TOP_K,
        pretrained = MOE_PRETRAINED,
        model_name = MOE_EXPERT_MODEL_NAME,
    )
    trainer_module = ResMoETrainerModule(model, optim.Adam(model.parameters(), lr = 1e-3))
    trainer = L.Trainer(
        default_root_dir = CHECKPOINT_DIR
    )
    dataset = get_dataset(data_dir = DATA_DIR, image_size = VIT_IMAGE_SIZE)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size = MOE_BATCH_SIZE, shuffle = True)
    trainer.fit(trainer_module, train_loader)
