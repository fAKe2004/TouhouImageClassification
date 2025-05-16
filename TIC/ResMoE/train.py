import os
import logging
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim
from tqdm import tqdm
import lightning as L
import lightning.pytorch.callbacks as callbacks

from TIC.ResMoE.model import MoEClassifier, make_ViTMoE
from TIC.utils.parameter import *
from TIC.utils.preprocess import get_dataset
from .parameter import *

logger = logging.getLogger(__name__)

def symmetric_cross_entropy(logits, targets, alpha=0.1, beta=1.0):
    ce = F.cross_entropy(logits, targets)
    rce = -torch.sum(F.softmax(logits, dim=1) * F.log_softmax(targets, dim=1), dim=1).mean()
    return alpha * ce + beta * rce

def load_balance_loss(gate_weights, top_k_indeces, num_experts):
    avg_expert_usages = torch.mean(gate_weights, dim = 0)
    return torch.matmul(gate_weights, avg_expert_usages.unsqueeze(1)).squeeze(1).mean()

def total_loss(logits, targets, gate_weights, top_k_indeces, alpha=0.5):
    assert not torch.isnan(logits).any(), "Logits contains NaN"
    assert torch.isfinite(logits).all(), "Logits contains Inf"

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

    def validation_step(self, batch, batch_idx):
        x, y = batch
        targets = F.one_hot(y, num_classes = self.model.num_classes).float()
        logits, gate_weights, top_k_indeces = self.model(x)
        balance_loss = load_balance_loss(gate_weights, top_k_indeces, gate_weights.shape[1])
        classification_loss = symmetric_cross_entropy(logits, targets)

        pred = torch.argmax(logits, dim = 1)
        acc = (pred == y).float().mean()
        self.log("val_balance_loss", balance_loss, on_epoch=True, prog_bar=True)
        self.log("val_classification_loss", classification_loss, on_epoch=True, prog_bar=True)
        self.log("val_accuracy", acc, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        targets = F.one_hot(y, num_classes = self.model.num_classes).float()
        logits, gate_weights, top_k_indeces = self.model(x)
        prediction = torch.argmax(logits, dim = 1)
        classification_loss = symmetric_cross_entropy(logits, targets)
        acc = (prediction == y).float().mean()
        self.log("test_classification_loss", classification_loss, on_epoch=True, prog_bar=True)
        self.log("test_accuracy", acc, on_epoch=True, prog_bar=True)

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

def get_model():
    return make_ViTMoE(
        num_classes = NUM_CLASSES,
        num_experts = MOE_NUM_EXPERTS,
        top_k = MOE_TOP_K,
        pretrained = MOE_PRETRAINED,
        model_name = MOE_EXPERT_MODEL_NAME,
    )

def get_trainer():
    checkpoint_callback_min = callbacks.ModelCheckpoint(
        monitor = "val_classification_loss",
        save_top_k = MOE_CHECKPOINT_MIN_K,
        mode = "min",
        dirpath = CHECKPOINT_DIR,
        filename = "checkpoint_ResMoE_{epoch:02d}_{val_classification_loss:.4f}",
    )
    check_point_call_back_last = callbacks.ModelCheckpoint(
        monitor = "epoch",
        save_top_k = MOE_CHECKPOINT_LAST_K,
        mode = "max",
        dirpath = CHECKPOINT_DIR,
        filename = "checkpoint_ResMoE_{epoch:02d}_{val_classification_loss:.4f}",
    )
    return L.Trainer(
        max_epochs = MOE_MAX_EPOCHS,
        limit_train_batches = MOE_LIMIT_TRAIN_BATCHES_PER_EPOCH,
        limit_val_batches = MOE_LIMIT_VAL_BATCHES_PER_EPOCH,
        default_root_dir = MOE_ROOT_DIR,
        callbacks = [checkpoint_callback_min, check_point_call_back_last],
        profiler = MOE_PROFILER,
        precision = MOE_ENABLE_AMP,
        accumulate_grad_batches = MOE_ACCUMULATE_GRAD_BATCHES,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Training script for ResMoE")
    parser.add_argument("--restore", "-r", type=str, default=None, help="Path to the checkpoint to restore, if none is given, will train from scratch")
    args = parser.parse_args()

    torch.set_float32_matmul_precision(MOE_TRAIN_PRECISION)

    model = get_model()
    trainer_module = ResMoETrainerModule(model, optim.Adam(model.parameters(), lr = 1e-3))
    trainer = get_trainer()
    dataset = get_dataset(data_dir = DATA_DIR, image_size = VIT_IMAGE_SIZE)
    testset = get_dataset(data_dir = TEST_DIR, image_size = VIT_IMAGE_SIZE)
    train_set_size = int(len(dataset) * MOE_TRAIN_SPLIT)
    val_set_size = len(dataset) - train_set_size

    seed = torch.Generator().manual_seed(42)
    train_set, valid_set = data.random_split(dataset, [train_set_size, val_set_size], generator = seed)

    train_loader = data.DataLoader(train_set, batch_size = MOE_BATCH_SIZE, shuffle = True)
    valid_loader = data.DataLoader(valid_set, batch_size = MOE_BATCH_SIZE, shuffle = False)
    test_loader = data.DataLoader(testset, batch_size = MOE_BATCH_SIZE, shuffle = False)
    trainer.fit(trainer_module, train_loader, valid_loader, ckpt_path = args.restore)
    trainer.test(trainer_module, test_loader)
