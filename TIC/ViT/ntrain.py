import os.path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.transforms import v2
import torchvision.datasets as datasets
import lightning as L
import lightning.pytorch.callbacks as Lc

from TIC.utils.parameter import *
from .model import ViT

class ViTLModule(L.LightningModule):

    def __init__(self, num_classes : int,
                 pretrianed : bool,
                 model_name : str,
                 lr : float,
                 weight_decay : float,
                 full_finetune: bool = True):
        super().__init__()
        self.num_classes = num_classes
        self.vit = ViT(num_classes, pretrianed, model_name)
        self.lr = lr
        self.weight_decay = weight_decay
        self.cutmix_or_mixup = v2.RandomChoice([
            v2.CutMix(num_classes = num_classes),
            v2.MixUp(num_classes = num_classes),
        ])
        if not full_finetune:
            for param in self.vit.base_model.parameters():
                param.requires_grad = False

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return optimizer

    def training_step(self, batch, batch_idx):
        x, y = batch
        x, y = self.cutmix_or_mixup(x, y)
        logits = self.vit(x).logits
        loss = F.cross_entropy(logits, y)
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.vit(x).logits
        loss = F.cross_entropy(logits, y)
        self.log('val_loss', loss, prog_bar=True)
        pred = logits.argmax(dim=1)
        acc = (pred == y).float().mean()
        self.log('val_acc', acc, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self.vit(x).logits
        pred = logits.argmax(dim=1)
        acc = (pred == y).float().mean()
        self.log('test_acc', acc, prog_bar=True)

class AugmentedDataset(L.LightningDataModule):

    def __init__(self, train_path : str = DATA_DIR, test_path : str = TEST_DIR, batch_size : int = 8, train_split : float = 0.8, num_workers : int = 8, image_size = VIT_IMAGE_SIZE):
        super().__init__()
        self.train_path = train_path
        self.test_path = test_path
        self.batch_size = batch_size
        self.image_size = image_size
        self.train_split = train_split

    def setup(self, stage : str):
        if stage == 'fit':
            transform = v2.Compose([
                v2.RandomResizedCrop(self.image_size),          
                v2.RandomHorizontalFlip(),         
                v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  
                v2.RandomGrayscale(p=0.2),
                v2.RandomErasing(p=0.5),
                v2.ToTensor(),                     
                v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            self.dataset = datasets.ImageFolder(self.train_path, transform = transform)
            train_size = int(len(self.dataset) * self.train_split)
            val_size = len(self.dataset) - train_size
            self.train_dataset, self.val_dataset = torch.utils.data.random_split(self.dataset, [train_size, val_size])

        if stage == 'test':
            transform = v2.Compose([
                v2.Resize(self.image_size),
                v2.ToTensor(),
                v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            self.test_dataset = datasets.ImageFolder(self.test_path, transform = transform)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=8)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=8)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=8)

if __name__ == '__main__':
    PRETRAINED = True
    MODEL_NAME = 'google/vit-large-patch16-224'
    LR = 1e-5
    WEIGHT_DECAY = 0.01
    FULL_FINETUNE = False
    BATCH_SIZE = 16
    NUM_WORKERS = 8
    TRAIN_SPLIT = 0.8
    TRAIN_ID = "nViT"
    DATA_DIR = "data_filtered_vit_base"

    torch.set_float32_matmul_precision('high')

    L.seed_everything(42)

    lmodel = ViTLModule(NUM_CLASSES, PRETRAINED, MODEL_NAME, LR, WEIGHT_DECAY, FULL_FINETUNE)
    data = AugmentedDataset(DATA_DIR, TEST_DIR, BATCH_SIZE, TRAIN_SPLIT, NUM_WORKERS)

    trainer = L.Trainer(
        max_epochs=10,
        callbacks = [
            Lc.ModelCheckpoint(
                monitor='val_acc', 
                mode='max', 
                save_top_k=3,
                dirpath = os.path.join(CHECKPOINT_DIR, TRAIN_ID),
                filename = "checkpoint_%s_{epoch:02d}_{val_acc:.4f}" % (TRAIN_ID),
            ),
            Lc.ModelCheckpoint(
                monitor = "epoch",
                mode = "max",
                save_top_k = 3,
                every_n_epochs = 3,
                dirpath = os.path.join(CHECKPOINT_DIR, TRAIN_ID),
                filename = "checkpoint_%s_{epoch:02d}_{val_acc:.4f}" % (TRAIN_ID),
            ),
            Lc.EarlyStopping(monitor='val_acc', mode='max', patience=3),
        ],
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        precision="bf16-mixed",
        default_root_dir = f"log/{TRAIN_ID}"
    )

    trainer.fit(lmodel, datamodule = data)
