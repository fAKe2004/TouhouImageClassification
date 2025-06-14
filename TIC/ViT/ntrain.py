import os.path
import argparse

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
                 pretrained : bool,
                 model_name : str,
                 lr : float,
                 weight_decay : float,
                 enable_mixup : bool = True,
                 full_finetune: bool = True):
        super().__init__()
        self.num_classes = num_classes
        self.vit = ViT(num_classes, pretrained, model_name)
        self.lr = lr
        self.weight_decay = weight_decay
        self.cutmix_or_mixup = v2.RandomChoice([
            v2.CutMix(num_classes = num_classes),
            v2.MixUp(num_classes = num_classes),
        ])
        self.enable_mixup = enable_mixup
        if not full_finetune:
            for param in self.vit.base_model.parameters():
                param.requires_grad = False

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return optimizer

    def training_step(self, batch, batch_idx):
        x, y = batch
        if self.enable_mixup:
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

    def __init__(self, 
                 train_path : str = DATA_DIR, 
                 test_path : str = TEST_DIR, 
                 batch_size : int = 8, 
                 train_split : float = 0.8,
                 num_workers : int = 8,
                 image_size = VIT_IMAGE_SIZE,
                 enable_augmentation : bool = True,
                 enable_diversity : bool = True,
                 enable_generalization : bool = True,
                 only_grey_augmentation : bool = False):
        super().__init__()
        self.train_path = train_path
        self.test_path = test_path
        self.batch_size = batch_size
        self.image_size = image_size
        self.train_split = train_split
        self.num_workers = num_workers
        self.enable_augmentation = enable_augmentation
        self.enable_diversity = enable_diversity
        self.enable_generalization = enable_generalization
        self.only_grey_augmentation = only_grey_augmentation

    def setup(self, stage : str):
        if stage == 'fit':
            if self.enable_augmentation:
                if self.only_grey_augmentation:
                    transform = v2.Compose([
                        v2.Resize(self.image_size),
                        v2.RandomGrayscale(p=0.2),
                        v2.ToTensor(),                     
                        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                    ])
                elif self.enable_diversity and self.enable_generalization:
                    transform = v2.Compose([
                        v2.RandomResizedCrop(self.image_size),          
                        v2.RandomHorizontalFlip(),         
                        v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  
                        v2.RandomGrayscale(p=0.2),
                        v2.RandomErasing(p=0.5),
                        v2.ToTensor(),                     
                        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                    ])
                elif self.enable_diversity:
                    transform = v2.Compose([
                        v2.Resize(self.image_size),          
                        v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  
                        v2.RandomGrayscale(p=0.2),
                        v2.ToTensor(),                     
                        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                    ])
                elif self.enable_generalization:
                    transform = v2.Compose([
                        v2.RandomResizedCrop(self.image_size),          
                        v2.RandomHorizontalFlip(),         
                        v2.RandomErasing(p=0.5),
                        v2.ToTensor(),                     
                        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                    ])
                else:
                    raise Exception("Must select diversity or generalization!")
            else:
                transform = v2.Compose([
                    v2.Resize(self.image_size),
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
        return torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

def train_main(
        PRETRAINED : bool,
        MODEL_NAME : str,
        LR : float,
        WEIGHT_DECAY : float,
        FULL_FINETUNE : bool,
        BATCH_SIZE : int,
        NUM_WORKERS : int,
        TRAIN_SPLIT : float,
        DATA_DIR : str,
        MAX_EPOCHS : int,
        ENABLE_MIX_UP : bool,
        ENABLE_AUGMENTATION : bool,
        TRAIN_ID : str,
        PATIENCE : int = 3,
        ONLY_GREY_AUGMENTATION : bool = False,
        ENABLE_DIVERSITY : bool = True,
        ENABLE_GENERALIZATION : bool = True,
):
    parser = argparse.ArgumentParser()
    parser.add_argument('--restore', '-r', type = str, default = None, help = 'Path to the checkpoint to restore')
    parser.add_argument('--test', '-t', action='store_true', help = 'Only test model without training')
    parser.add_argument('--transform', '-tr', type = str, default = None, help = 'Transform the checkpoint')
    args = parser.parse_args()

    torch.set_float32_matmul_precision('high')

    L.seed_everything(42)

    if args.transform:
        if not args.restore:
            print("No checkpoint to transform")
            exit(-1)
        lmodel = ViTLModule.load_from_checkpoint(args.restore, num_classes = NUM_CLASSES, pretrained = PRETRAINED, model_name = MODEL_NAME, lr = LR, weight_decay = WEIGHT_DECAY, enable_mixup=ENABLE_MIX_UP, full_finetune=FULL_FINETUNE)
        torch.save(lmodel.vit.state_dict(), args.transform)
        exit(0)

    lmodel = ViTLModule(
        num_classes = NUM_CLASSES, 
        pretrained = PRETRAINED, 
        model_name = MODEL_NAME, 
        lr = LR, 
        weight_decay = WEIGHT_DECAY, 
        enable_mixup=ENABLE_MIX_UP, 
        full_finetune=FULL_FINETUNE,
    )

    data = AugmentedDataset(
        train_path = DATA_DIR, 
        test_path=TEST_DIR, 
        batch_size=BATCH_SIZE, 
        train_split=TRAIN_SPLIT, 
        num_workers=NUM_WORKERS, 
        image_size=VIT_IMAGE_SIZE, 
        enable_augmentation=ENABLE_AUGMENTATION,
        enable_diversity=ENABLE_DIVERSITY,
        enable_generalization=ENABLE_GENERALIZATION,
        only_grey_augmentation=ONLY_GREY_AUGMENTATION,
    )

    trainer = L.Trainer(
        max_epochs=MAX_EPOCHS,
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
            Lc.EarlyStopping(monitor='val_acc', mode='max', patience=PATIENCE) if PATIENCE > 0 else Lc.Callback(),
        ],
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        precision="bf16-mixed",
        default_root_dir = f"log/{TRAIN_ID}"
    )

    if not args.test:
        trainer.fit(lmodel, datamodule = data, ckpt_path = args.restore)
    
    trainer.test(lmodel, datamodule = data, ckpt_path = args.restore if args.test else None)

if __name__ == '__main__':
    train_main(
        PRETRAINED = True,
        MODEL_NAME = 'google/vit-large-patch16-224',
        LR = 1e-5,
        WEIGHT_DECAY = 0.01,
        FULL_FINETUNE = True,
        BATCH_SIZE = 8,
        NUM_WORKERS = 4,
        TRAIN_SPLIT = 0.8,
        TRAIN_ID = "nViT",
        DATA_DIR = "data_filtered_vit_base",
        MAX_EPOCHS = 20,
        ENABLE_MIX_UP = True,
        ENABLE_AUGMENTATION = True,
    )
