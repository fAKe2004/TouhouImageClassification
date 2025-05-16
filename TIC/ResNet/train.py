import torch
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import math
import logging
import os
import sys
from typing import Optional

from TIC.utils.preprocess import get_dataset
from TIC.utils.parameter import *
from TIC.ResNet.model import resnet152, resnet18
from TIC.utils.loss_function import SymmetricCrossEntropyLoss

def get_logger(name, log_dir='log'):
  """
  Configures and returns a logger.

  Args:
      name (str): Name for the logger and the log file.
      log_dir (str): Directory to save the log file. Defaults to 'log'.

  Returns:
      logging.Logger: Configured logger instance.
  """
  os.makedirs(log_dir, exist_ok=True)
  
  logger = logging.getLogger(name)
  logger.setLevel(logging.INFO)

  if not logger.handlers:
    log_path = os.path.join(log_dir, f"{name}.log")
    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.INFO)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
  return logger

def train_step(model, data, optimizer, criterion, scaler):
  model.train()
  optimizer.zero_grad()
  with torch.autocast(device_type='cuda', dtype=torch.float16):
    inputs, labels = map(lambda x: x.to("cuda"), data)
    outputs = model(inputs)
    loss = criterion(outputs, labels)
  scaler.scale(loss).backward()
  scaler.step(optimizer)
  scaler.update()
  return loss.item()

def validate_step(model, data, criterion):
  model.eval()
  with torch.no_grad() and torch.amp.autocast("cuda"):
    inputs, labels = map(lambda x: x.to("cuda"), data)
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    correct = (outputs.argmax(dim=1) == labels).sum().item()
  return loss.item(), correct

def early_exit(timeline, max_tolerant_epoch, logger):
  if len(timeline) < max_tolerant_epoch:
      return False

  relevant_window = timeline[-(max_tolerant_epoch+1):]
  best_in_window_start = relevant_window[0]
  worse_in_tolerant_period = all(loss >= best_in_window_start for loss in relevant_window[1:])

  if worse_in_tolerant_period:
      logger.info(f"Validation loss has not improved for {max_tolerant_epoch} epochs. Stopping training.")
      return True
  
  return False

def train_model(model: torch.nn.Module, 
                dataset: torch.utils.data.Dataset, 
                optimizer: torch.optim.Optimizer, 
                scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
                criterion: torch.nn.Module, 
                batch_size: int, 
                num_epochs: int, 
                max_tolerant_epoch: int,
                save_path: str,
                logger: logging.Logger,
                skip_optimizer_load: bool = False,
                ):
  
  latest_epoch_found = 0
  for i in range(num_epochs - 1, 0, -1):
      potential_path = save_path.format(epoch=i)
      if os.path.exists(potential_path):
          latest_epoch_found = i
          break

  if latest_epoch_found > 0:
      logger.info(f"Resuming from epoch {latest_epoch_found + 1}")
      ckpt = torch.load(save_path.format(epoch=latest_epoch_found), weights_only=False)
      if isinstance(ckpt, tuple):
        model_state_dict, optimizer_state_dict = ckpt
        model.load_state_dict(model_state_dict)
        if not skip_optimizer_load:
          optimizer.load_state_dict(optimizer_state_dict)
        elif scheduler:
          # resume lr for the epoch with scheduler
          optimizer.step()
          for i in range(latest_epoch_found):
            scheduler.step()
            
      else:
        model.load_state_dict(ckpt)
  else:
      logger.info("Starting training from scratch.")
  
  
  
  
  dataset_size = len(dataset)
  val_size = dataset_size // 10
  train_size = dataset_size - val_size
  torch.manual_seed(0) # ensure split consistency across runs
  train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

  train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=8)
  val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=8)

  avg_val_loss_timeline = []

  scaler = torch.GradScaler()
  def train_loop(epoch):
    if scheduler:
      logger.info(f"Learning rate for next epoch: {scheduler.get_last_lr()[0]:.6f}")
    model.train()
    running_train_loss = 0.0
    last_nan_iter = None
    train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]", file=sys.stdout)
    for i, data in enumerate(train_pbar):
      loss = train_step(model, data, optimizer, criterion, scaler)
      
      if math.isnan(loss):
        loss = running_train_loss / (i + 1)
        last_nan_iter = i
      
      running_train_loss += loss
        
      current_avg_loss = running_train_loss / (i + 1)
      
      train_pbar.set_postfix({'loss': f'{current_avg_loss:.4f}', 'last_nan_iter': last_nan_iter})
    avg_train_loss = running_train_loss / len(train_loader) if len(train_loader) > 0 else 0.0
    return avg_train_loss

  def validate_loop(epoch):
    running_val_loss = 0.0
    last_nan_iter = None
    
    correct_sample = 0
    total_sample = 0
  
    val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]", file=sys.stdout)
    for i, data in enumerate(val_pbar):
      loss, correct = validate_step(model, data, criterion)
      
      if math.isnan(loss):
        loss = running_val_loss / (i + 1)
        last_nan_iter = i

      running_val_loss += loss      
      correct_sample += correct
      total_sample += len(data[1])
      current_avg_loss = running_val_loss / (i + 1)
      val_pbar.set_postfix({'val_loss': f'{current_avg_loss:.4f}', 'last_nan_iter': last_nan_iter})
      
    avg_val_loss = running_val_loss / len(val_loader)
    accuracy = (correct_sample / total_sample * 100) if total_sample > 0 else 0.0
    return avg_val_loss, accuracy

  if latest_epoch_found != 0:
    logger.info("Validate before training:")
    avg_val_loss, accuracy = validate_loop(latest_epoch_found - 1)
    logger.info(f'Epoch [Before Training], Validation Loss: {avg_val_loss:.4f}, Accuracy: {accuracy:.2f}%')
    
  # training by epoch
  for epoch in range(latest_epoch_found, num_epochs):
    avg_train_loss = train_loop(epoch)
    
    avg_val_loss, accuracy = validate_loop(epoch)
    
    avg_val_loss_timeline.append(avg_val_loss)
    
    os.makedirs(os.path.dirname(save_path.format(epoch=epoch)), exist_ok=True)
    torch.save(
      (model.state_dict(), optimizer.state_dict()), 
       save_path.format(epoch=epoch+1))
    
    logger.info(f"Checkpoint saved to {save_path.format(epoch=epoch)}")
    logger.info(f'Epoch [{epoch+1}/{num_epochs}], Training Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}, Accuracy: {accuracy:.2f}%')
    
    if early_exit(avg_val_loss_timeline, max_tolerant_epoch, logger):
      break
    
    if scheduler:
      scheduler.step()

if __name__ == '__main__':
  # Hyperparameters
  
  BATCH_SIZE = 80
  NUM_EPOCHS = 25
  LR = 5e-2
  SKIP_OPTIMIZER_LOAD = True # set to true if you found previous scheduler is not proper
  
  os.makedirs(CHECKPOINT_DIR, exist_ok=True)
  
  SAVE_PATH = os.path.join(CHECKPOINT_DIR, 'ResNet_model_{epoch}.pth')
  MAX_TOLERANT_EPOCH = 3
  LOG_NAME = 'ResNet_train'

  logger = get_logger(LOG_NAME, LOG_DIR)

  logger.info("Starting training script.")
  logger.info(f"Parameters: BATCH_SIZE={BATCH_SIZE}, IMAGE_SIZE={IMAGE_SIZE}, NUM_EPOCHS={NUM_EPOCHS}, MAX_TOLERANT_EPOCH={MAX_TOLERANT_EPOCH}")

  torch.set_default_dtype(torch.float32)

  logger.info(f"Loading dataset from {DATA_DIR}...")
  dataset = get_dataset( 
      data_dir=DATA_DIR,
      image_size=IMAGE_SIZE,
  )
  num_classes = len(dataset.classes)
  logger.info(f"Dataset loaded. Number of classes: {num_classes}")
  
  model = resnet152(num_classes=num_classes).to("cuda")
  optimizer = torch.optim.SGD(model.parameters(), lr=LR)
  scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.25)
  # criterion = torch.nn.CrossEntropyLoss()
  criterion = SymmetricCrossEntropyLoss()
  

  logger.info("Starting model training...")
  train_model(model, dataset, optimizer, scheduler, criterion,
              num_epochs=NUM_EPOCHS, 
              batch_size=BATCH_SIZE,
              max_tolerant_epoch=MAX_TOLERANT_EPOCH,
              save_path=SAVE_PATH,
              logger=logger,
              skip_optimizer_load=True,
              )
              
  logger.info("Training finished.")