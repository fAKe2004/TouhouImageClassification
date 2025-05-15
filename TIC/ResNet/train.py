import torch
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import math
import logging
import os
import sys
from typing import Optional
import argparse

from TIC.utils.preprocess import get_dataset
from TIC.utils.parameter import (
    CHECKPOINT_DIR as DEFAULT_CHECKPOINT_DIR,
    LOG_DIR as DEFAULT_LOG_DIR,
    DATA_DIR as DEFAULT_DATA_DIR,
    IMAGE_SIZE as DEFAULT_IMAGE_SIZE
)
from TIC.ResNet.model import resnet152, resnet18

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
  latest_scheduler_state = None

  for i in range(num_epochs, 0, -1): 
      potential_path = save_path.format(epoch=i)
      if os.path.exists(potential_path):
          latest_epoch_found = i
          break

  start_epoch = 0
  if latest_epoch_found > 0:
      logger.info(f"Resuming from checkpoint for epoch {latest_epoch_found}")
      checkpoint_path = save_path.format(epoch=latest_epoch_found)
      ckpt = torch.load(checkpoint_path, map_location='cuda')
      
      model_state_dict = None
      optimizer_state_dict = None

      if isinstance(ckpt, tuple):
          if len(ckpt) >= 2:
              model_state_dict = ckpt[0]
              optimizer_state_dict = ckpt[1]
          if len(ckpt) >= 3:
              latest_scheduler_state = ckpt[2]
      else:
          model_state_dict = ckpt

      if model_state_dict:
          model.load_state_dict(model_state_dict)
      
      if not skip_optimizer_load:
          if optimizer_state_dict:
              optimizer.load_state_dict(optimizer_state_dict)
              logger.info("Loaded optimizer state.")
          if scheduler and latest_scheduler_state:
              scheduler.load_state_dict(latest_scheduler_state)
              logger.info("Loaded scheduler state.")
          elif scheduler and not latest_scheduler_state:
              logger.warning("Optimizer loaded, but no scheduler state found in checkpoint. Scheduler might not be in sync if it's not StepLR-like.")
              logger.info(f"Advancing scheduler for {latest_epoch_found} completed epochs as a fallback.")
              for _ in range(latest_epoch_found):
                  scheduler.step()
      elif scheduler:
          logger.info(f"Skipping optimizer load. Manually advancing scheduler to epoch {latest_epoch_found}")
          for _ in range(latest_epoch_found):
              scheduler.step()
      
      start_epoch = latest_epoch_found
      logger.info(f"Model, optimizer, and scheduler states (if applicable) loaded. Resuming training from epoch {start_epoch}.")

  else:
      logger.info("Starting training from scratch.")
      start_epoch = 0
  
  dataset_size = len(dataset)
  val_size = dataset_size // 10
  train_size = dataset_size - val_size
  torch.manual_seed(0)
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
    
  for epoch in range(start_epoch, num_epochs):
    avg_train_loss = train_loop(epoch)
    
    avg_val_loss, accuracy = validate_loop(epoch)
    
    avg_val_loss_timeline.append(avg_val_loss)
    
    current_epoch_save_path = save_path.format(epoch=epoch + 1)
    os.makedirs(os.path.dirname(current_epoch_save_path), exist_ok=True)
    
    checkpoint_data = (
        model.state_dict(),
        optimizer.state_dict()
    )
    if scheduler:
        checkpoint_data += (scheduler.state_dict(),)
        
    torch.save(checkpoint_data, current_epoch_save_path)
    
    logger.info(f"Checkpoint saved to {current_epoch_save_path}")
    logger.info(f'Epoch [{epoch+1}/{num_epochs}], Training Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}, Accuracy: {accuracy:.2f}%')
    
    if early_exit(avg_val_loss_timeline, max_tolerant_epoch, logger):
      break
    
    if scheduler:
      scheduler.step()

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description="Train a ResNet model.")

  parser.add_argument('--model-type', type=str, default='resnet152', choices=['resnet18', 'resnet152'], help="Type of ResNet model to train.")
  parser.add_argument('--num-epochs', type=int, default=25, help='Number of training epochs.')
  parser.add_argument('--batch-size', type=int, default=80, help='Batch size for training and validation.')
  parser.add_argument('--lr', type=float, default=5e-2, help='Initial learning rate for SGD optimizer.')
  
  parser.add_argument('--scheduler-step-size', type=int, default=5, help='Step size for StepLR scheduler.')
  parser.add_argument('--scheduler-gamma', type=float, default=0.25, help='Gamma factor for StepLR scheduler.')

  parser.add_argument('--load-optimizer-state', action='store_true', help='Load optimizer and scheduler state when resuming (if available). If not set, only model weights are loaded and scheduler is advanced manually.')
  
  parser.add_argument('--max-tolerant-epoch', type=int, default=3, help='Max tolerant epochs for early stopping.')

  parser.add_argument('--data-dir', type=str, default=DEFAULT_DATA_DIR, help='Directory containing the dataset.')
  parser.add_argument('--image-size', type=int, default=DEFAULT_IMAGE_SIZE, help='Image size for preprocessing.')
  parser.add_argument('--checkpoint-dir', type=str, default=DEFAULT_CHECKPOINT_DIR, help='Directory to save model checkpoints.')
  parser.add_argument('--checkpoint-name-prefix', type=str, default='ResNet_model', help='Prefix for checkpoint filenames.')
  parser.add_argument('--log-dir', type=str, default=DEFAULT_LOG_DIR, help='Directory to save logs.')
  parser.add_argument('--log-name', type=str, default='ResNet_train', help='Name for the log file.')

  args = parser.parse_args()

  skip_optimizer_load_val = not args.load_optimizer_state

  os.makedirs(args.checkpoint_dir, exist_ok=True)
  save_path_template = os.path.join(args.checkpoint_dir, f'{args.checkpoint_name_prefix}_{{epoch}}.pth')
  
  logger = get_logger(args.log_name, args.log_dir)

  logger.info("Starting ResNet training script with parsed arguments.")
  logger.info(f"Full configuration: {vars(args)}")
  logger.info(f"Effective skip_optimizer_load: {skip_optimizer_load_val}")

  torch.set_default_dtype(torch.float32)

  logger.info(f"Loading dataset from {args.data_dir} with image size {args.image_size}...")
  dataset = get_dataset( 
      data_dir=args.data_dir,
      image_size=args.image_size,
  )
  num_classes = len(dataset.classes)
  logger.info(f"Dataset loaded. Number of classes: {num_classes}")
  
  if args.model_type == 'resnet152':
    model = resnet152(num_classes=num_classes).to("cuda")
  elif args.model_type == 'resnet18':
    model = resnet18(num_classes=num_classes).to("cuda")
  else:
    logger.error(f"Unsupported model type: {args.model_type}")
    sys.exit(1)
  logger.info(f"Using model: {args.model_type}")
    
  optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
  scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.scheduler_step_size, gamma=args.scheduler_gamma)
  criterion = torch.nn.CrossEntropyLoss()
  
  logger.info("Starting model training...")
  train_model(model, dataset, optimizer, scheduler, criterion,
              num_epochs=args.num_epochs, 
              batch_size=args.batch_size,
              max_tolerant_epoch=args.max_tolerant_epoch,
              save_path=save_path_template,
              logger=logger,
              skip_optimizer_load=skip_optimizer_load_val,
              )
              
  logger.info("Training finished.")