import csv

import sklearn.preprocessing as preprocessing
import torch
import torch.utils.data as data
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.datasets as vdatasets
import lightning as L
import lightning.pytorch.callbacks as callbacks

from TIC.utils.parameter import *
from TIC.utils.preprocess import get_dataset, get_transforms
from .parameter import *
from . import model as TreeViT

def symmetric_cross_entropy(logits, targets, alpha=0.1, beta=1.0):
    ce = F.cross_entropy(logits, targets)
    rce = -torch.sum(F.softmax(logits, dim=1) * F.log_softmax(targets, dim=1), dim=1).mean()
    return alpha * ce + beta * rce

class CommonTrainerModule(L.LightningModule):
    def __init__(self, model : nn.Module, optimizer : optim.Optimizer, slogan : str):
        super().__init__()
        self.model = model
        self.optimizer = optimizer

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model(x)
        loss = symmetric_cross_entropy(logits, y)
        self.log(f"{self.slogan}_train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model(x)
        loss = symmetric_cross_entropy(logits, y)
        self.log(f"{self.slogan}_root_val_loss", loss, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model(x)
        pred = logits.argmax(dim = 1)
        loss = symmetric_cross_entropy(logits, y)
        acc = (pred == y).float().mean()
        self.log(f"{self.slogan}_test_loss", loss, on_epoch=True, prog_bar=True)
        self.log(f"{self.slogan}_test_acc", acc, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        return self.optimizer


class CategoryDataset(data.Dataset):

    '''
    Generate (image, category) datasets
    '''

    def __init__(self, image_dataset : vdatasets.ImageFolder, map_indeces_dict : dict):
        self.image_dataset = image_dataset
        self.map_indeces_dict = map_indeces_dict

    def __getitem__(self, index):
        x, y = self.image_dataset[index]
        return x, self.map_indeces_dict[y]

    def __len__(self):
        return len(self.image_dataset)

def get_category_labeler(map_dict : dict) -> preprocessing.LabelEncoder:
    le = preprocessing.LabelEncoder()
    le.fit(map_dict.values())
    return le

def get_partial_dataset(target_category : int, image_dataset : vdatasets.ImageFolder, map_labeled_dict : dict, le : preprocessing.LabelEncoder):
    indeces = list(filter(lambda y : map_labeled_dict[y] == target_category, range(len(image_dataset.classes))))
    return data.Subset(image_dataset, indeces)

def load_map_dict(filename : str) -> dict:
    '''
    Load the classes to category mapping.
    '''
    map_dict = {}
    with open(filename, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            map_dict[row["class"]] = row["category"]
    return map_dict

def make_map_indeces_dict(map_dict : dict, class_to_idx : dict, category_to_idx : preprocessing.LabelEncoder) -> dict:
    '''
    Make the map from classes to category, but in the form of int -> int
    '''
    return {class_to_idx[k] : category_to_idx.transform(v)[0] for k, v in map_dict.items()}

def get_model(map_indeces_dict : dict):
    return TreeViT.make_TreeViT(
        num_categories = len(set(map_indeces_dict.values())),
        num_classes = NUM_CLASSES,
        top_k = TREEVIT_TOP_K,
        sons_pretrained = TREEVIT_SONS_PRETRAINED,
        root_pretrained = TREEVIT_ROOT_PRETRAINED,
    )

def get_trainer(slogan : str, max_epochs : int | None = None):
    checkpoint_callback_min = callbacks.ModelCheckpoint(
        monitor = "{slogan}_val_loss",
        save_top_k = TREEVIT_CHECKPOINT_MIN_K,
        mode = "min",
        dirpath = CHECKPOINT_DIR,
        filename = "checkpoint_TreeViT_{slogan}_{epoch:02d}_{val_classification_loss:.4f}",
    )
    check_point_call_back_last = callbacks.ModelCheckpoint(
        monitor = "epoch",
        save_top_k = TREEVIT_CHECKPOINT_LAST_K,
        mode = "max",
        dirpath = CHECKPOINT_DIR,
        filename = "checkpoint_ResMoE_{slogan}_{epoch:02d}_{val_classification_loss:.4f}",
    )
    return L.Trainer(
        max_epochs = max_epochs if max_epochs is not None else TREEVIT_MAX_EPOCHS,
        limit_train_batches = TREEVIT_LIMIT_TRAIN_BATCHES_PER_EPOCH,
        limit_val_batches = TREEVIT_LIMIT_VAL_BATCHES_PER_EPOCH,
        default_root_dir = TREEVIT_ROOT_DIR,
        callbacks = [checkpoint_callback_min, check_point_call_back_last],
        profiler = TREEVIT_PROFILER,
        precision = TREEVIT_ENABLE_AMP,
        accumulate_grad_batches = TREEVIT_ACCUMULATE_GRAD_BATCHES,
    )

def auto_train(model : nn.Module, slogan : str, dataset : data.Dataset, testset : data.Dataset, restore : str | None = None, max_epochs : int | None = None):
    trainer = get_trainer(slogan = slogan, max_epochs = max_epochs)
    train_size = int(len(dataset) * TREEVIT_TRAIN_SET_SIZE)
    val_size = len(dataset) - train_size
    train_set, val_set = data.random_split(dataset, [train_size, val_size])
    trainer_module = CommonTrainerModule(model, optim.Adam(model.parameters(), lr = 1e-3), slogan)
    trainer.fit(trainer_module, train_set, val_set, ckpt_path = restore)
    trainer.test(trainer_module, testset)

def train_root(model : TreeViT.TreeModule, restore : str = None):
    dataset = get_dataset(data_dir = UNFILTERED_DATA_DIR, image_size = VIT_IMAGE_SIZE)
    testset = get_dataset(data_dir = TEST_DIR, image_size = VIT_IMAGE_SIZE)
    map_dict = load_map_dict(TREEVIT_MAP_FILE)
    map_indeces_dict = make_map_indeces_dict(map_dict, dataset.class_to_idx, get_category_labeler(map_dict))
    category_dataset = CategoryDataset(dataset, map_indeces_dict)
    category_testset = CategoryDataset(testset, map_indeces_dict)
    auto_train(model.root, "root", category_dataset, category_testset, restore)

def train_son(model : TreeViT.TreeModule, target_category : int, restore : str = None):
    dataset = get_dataset(data_dir = UNFILTERED_DATA_DIR, image_size = VIT_IMAGE_SIZE)
    testset = get_dataset(data_dir = TEST_DIR, image_size = VIT_IMAGE_SIZE)
    map_dict = load_map_dict(TREEVIT_MAP_FILE)
    map_indeces_dict = make_map_indeces_dict(map_dict, dataset.class_to_idx, get_category_labeler(map_dict))
    son_dataset = get_partial_dataset(target_category, dataset, map_indeces_dict, get_category_labeler(map_dict))
    son_testset = get_partial_dataset(target_category, testset, map_indeces_dict, get_category_labeler(map_dict))
    auto_train(model.sons[target_category], f"son_{target_category}", son_dataset, son_testset, restore)

def train_full(model : TreeViT.TreeModule, restore : str = None):
    dataset = get_dataset(data_dir = UNFILTERED_DATA_DIR, image_size = VIT_IMAGE_SIZE)
    testset = get_dataset(data_dir = TEST_DIR, image_size = VIT_IMAGE_SIZE)
    auto_train(model, "full", dataset, testset, restore, TREEVIT_MAX_EPOCHS)

if __name__ == "__main__":
    L.seed_everything(TREEVIT_RAND_SEED)

    map_dict = load_map_dict(TREEVIT_MAP_FILE)
    dataset = get_dataset(data_dir = UNFILTERED_DATA_DIR, image_size = VIT_IMAGE_SIZE)
    map_indeces_dict = make_map_indeces_dict(map_dict, dataset.class_to_idx, get_category_labeler(map_dict))
    model = get_model(map_indeces_dict)

    train_root(model)
    category_num = len(set(map_indeces_dict.values()))
    print("Category num:", category_num)
    for i in range(category_num):
        train_son(model, i)

    train_full(model)
