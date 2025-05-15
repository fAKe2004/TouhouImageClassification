"""
nohup python -m TIC.ViT.train > train.out 2>&1 &
"""

import torch
from torch.utils.data import DataLoader, random_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from tqdm import tqdm
import math
import logging
import os
import sys
from typing import Optional
import argparse
from transformers import get_linear_schedule_with_warmup

from TIC.utils.preprocess import get_dataset
from TIC.utils.parameter import (
    CHECKPOINT_DIR as DEFAULT_CHECKPOINT_DIR,
    LOG_DIR as DEFAULT_LOG_DIR,
    DATA_DIR as DEFAULT_DATA_DIR,
    VIT_IMAGE_SIZE as DEFAULT_VIT_IMAGE_SIZE
)
from TIC.ViT.model import ViT

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

def train_step(model, data, optimizer, criterion, scaler, scheduler=None):
  model.train()
  optimizer.zero_grad()
  with torch.autocast(device_type='cuda', dtype=torch.float16):
    inputs, labels = map(lambda x: x.to("cuda"), data)
    outputs = model(inputs)
    logits = outputs.logits
    loss = criterion(logits, labels)
  scaler.scale(loss).backward()
  scaler.step(optimizer)
  scaler.update()
  if scheduler:
      scheduler.step()
  return loss.item()

def validate_step(model, data, criterion):
  model.eval()
  with torch.no_grad() and torch.autocast(device_type='cuda', dtype=torch.float16):
    inputs, labels = map(lambda x: x.to("cuda"), data)
    outputs = model(inputs)
    logits = outputs.logits
    loss = criterion(logits, labels)
    correct = (logits.argmax(dim=1) == labels).sum().item()
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
                scheduler_per_epoch: bool = True
                ):

  latest_epoch_found = 0
  latest_scheduler_state = None
  for i in range(num_epochs, 0, -1):
      potential_path = save_path.format(epoch=i)
      if os.path.exists(potential_path):
          latest_epoch_found = i
          break

  if latest_epoch_found > 0:
      logger.info(f"Resuming from epoch {latest_epoch_found}")
      ckpt = torch.load(save_path.format(epoch=latest_epoch_found), map_location='cuda')
      if isinstance(ckpt, tuple) and len(ckpt) >= 2:
        model_state_dict, optimizer_state_dict = ckpt[0], ckpt[1]
        if len(ckpt) > 2:
            latest_scheduler_state = ckpt[2]

        model.load_state_dict(model_state_dict)
        if not skip_optimizer_load:
          optimizer.load_state_dict(optimizer_state_dict)
          if scheduler and latest_scheduler_state and scheduler_per_epoch:
              scheduler.load_state_dict(latest_scheduler_state)
              logger.info("Loaded scheduler state.")
          elif scheduler and not scheduler_per_epoch:
              logger.warning("Resuming per-step scheduler state not fully implemented, may restart LR schedule.")

        elif scheduler and scheduler_per_epoch:
          logger.info(f"Skipping optimizer load, manually advancing scheduler to epoch {latest_epoch_found}")
          for _ in range(latest_epoch_found):
              scheduler.step()

      else:
        model.load_state_dict(ckpt)
        logger.warning("Loaded checkpoint only contains model state_dict. Optimizer and scheduler state not loaded.")
      start_epoch = latest_epoch_found
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
    if scheduler and scheduler_per_epoch:
      logger.info(f"LR for epoch {epoch + 1}: {scheduler.get_last_lr()[0]:.6e}")
    elif scheduler and not scheduler_per_epoch:
        pass

    model.train()
    running_train_loss = 0.0
    last_nan_iter = None
    train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]", file=sys.stdout)
    for i, data in enumerate(train_pbar):
      current_scheduler = scheduler if not scheduler_per_epoch else None
      loss = train_step(model, data, optimizer, criterion, scaler, current_scheduler)

      if math.isnan(loss):
        logger.warning(f"NaN loss detected at training step {i} in epoch {epoch+1}. Replacing with avg loss.")
        loss = running_train_loss / (i + 1) if i > 0 else 0.0
        last_nan_iter = i

      running_train_loss += loss
      current_avg_loss = running_train_loss / (i + 1)

      postfix_dict = {'loss': f'{current_avg_loss:.4f}'}
      if last_nan_iter is not None:
          postfix_dict['last_nan_iter'] = last_nan_iter
      if scheduler and not scheduler_per_epoch:
          postfix_dict['lr'] = f'{optimizer.param_groups[0]["lr"]:.2e}'

      train_pbar.set_postfix(postfix_dict)

    avg_train_loss = running_train_loss / len(train_loader) if len(train_loader) > 0 else 0.0
    return avg_train_loss

  def validate_loop(epoch):
    optimizer.zero_grad(set_to_none=True)
    
    running_val_loss = 0.0
    last_nan_iter = None
    correct_sample = 0
    total_sample = 0

    val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]", file=sys.stdout)
    for i, data in enumerate(val_pbar):
      loss, correct = validate_step(model, data, criterion)

      if math.isnan(loss):
        logger.warning(f"NaN loss detected during validation step {i} in epoch {epoch+1}. Replacing with avg loss.")
        loss = running_val_loss / (i + 1) if i > 0 else 0.0
        last_nan_iter = i

      running_val_loss += loss
      correct_sample += correct
      total_sample += len(data[1])
      current_avg_loss = running_val_loss / (i + 1)

      postfix_dict = {'val_loss': f'{current_avg_loss:.4f}'}
      if last_nan_iter is not None:
          postfix_dict['last_nan_iter'] = last_nan_iter
      val_pbar.set_postfix(postfix_dict)

    avg_val_loss = running_val_loss / len(val_loader) if len(val_loader) > 0 else 0.0
    accuracy = (correct_sample / total_sample * 100) if total_sample > 0 else 0.0
    return avg_val_loss, accuracy

  if start_epoch > 0:
    logger.info(f"Validating model from loaded checkpoint (Epoch {start_epoch}) before resuming training...")
    avg_val_loss, accuracy = validate_loop(start_epoch - 1)
    logger.info(f'Epoch [{start_epoch}], Validation Loss: {avg_val_loss:.4f}, Accuracy: {accuracy:.2f}%')

  for epoch in range(start_epoch, num_epochs):
    avg_train_loss = train_loop(epoch)
    avg_val_loss, accuracy = validate_loop(epoch)
    avg_val_loss_timeline.append(avg_val_loss)

    checkpoint_data = (
        model.state_dict(),
        optimizer.state_dict()
    )
    if scheduler and scheduler_per_epoch:
        checkpoint_data += (scheduler.state_dict(),)

    current_epoch_save_path = save_path.format(epoch=epoch + 1)
    os.makedirs(os.path.dirname(current_epoch_save_path), exist_ok=True)
    torch.save(checkpoint_data, current_epoch_save_path)

    logger.info(f"Checkpoint saved to {current_epoch_save_path}")
    logger.info(f'Epoch [{epoch+1}/{num_epochs}], Training Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}, Accuracy: {accuracy:.2f}%')

    if early_exit(avg_val_loss_timeline, max_tolerant_epoch, logger):
      break

    if scheduler and scheduler_per_epoch:
      if isinstance(scheduler, ReduceLROnPlateau):
        scheduler.step(avg_val_loss)
      else:
        scheduler.step()


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description="Fine-tune a Vision Transformer (ViT) model.")

  parser.add_argument('--num-epochs', type=int, default=40, help='Number of training epochs.')
  parser.add_argument('--batch-size', type=int, default=30, help='Batch size for training and validation.')
  parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate.')
  parser.add_argument('--weight-decay', type=float, default=0.01, help='Weight decay for AdamW optimizer.')
  parser.add_argument('--skip_optimizer_load', action='store_true', help='Skip loading optimizer/scheduler state when resuming.')
  
  parser.add_argument('--model-name', type=str, default="google/vit-large-patch16-224-in21k", help='Name of the pretrained ViT model from Hugging Face.')
  parser.add_argument('--no-use-pretrained', action='store_false', dest='use_pretrained', help="Don't use pretrained ViT weights.")
  parser.set_defaults(use_pretrained=True)

  parser.add_argument('--scheduler-type', type=str, default='plateau', choices=['linear', 'plateau', 'none'], help='Type of learning rate scheduler.')
  parser.add_argument('--warmup-steps', type=int, default=500, help='Number of warmup steps for the linear scheduler.')
  parser.add_argument('--plateau-factor', type=float, default=0.1, help='Factor by which the learning rate will be reduced for ReduceLROnPlateau scheduler.')
  parser.add_argument('--plateau-patience', type=int, default=10, help='Number of epochs with no improvement after which learning rate will be reduced for ReduceLROnPlateau.')

  parser.add_argument('--max-tolerant-epoch', type=int, default=None, help='Max tolerant epochs for early stopping. If not set, defaults to num_epochs (disabling early stopping).')

  parser.add_argument('--data-dir', type=str, default=DEFAULT_DATA_DIR, help='Directory containing the dataset.')
  parser.add_argument('--image-size', type=int, default=DEFAULT_VIT_IMAGE_SIZE, help='Image size for ViT model.')
  parser.add_argument('--checkpoint-dir', type=str, default=DEFAULT_CHECKPOINT_DIR, help='Directory to save model checkpoints.')
  parser.add_argument('--checkpoint-name', type=str, default='ViT_model_finetune', help='Name prefix for the checkpoint file.')
  parser.add_argument('--log-dir', type=str, default=DEFAULT_LOG_DIR, help='Directory to save logs.')
  parser.add_argument('--log-name', type=str, default='ViT_finetune', help='Name for the log file.')
  

  args = parser.parse_args()

  max_tolerant_epoch_val = args.max_tolerant_epoch if args.max_tolerant_epoch is not None else args.num_epochs

  os.makedirs(args.checkpoint_dir, exist_ok=True)
  save_path_template = os.path.join(args.checkpoint_dir, args.checkpoint_name+'_{epoch}.pth')
  
  logger = get_logger(args.log_name, args.log_dir)

  logger.info("Starting ViT fine-tuning script with parsed arguments.")
  logger.info(f"Full configuration: {vars(args)}")
  
  torch.set_default_dtype(torch.float32)

  logger.info(f"Loading dataset from {args.data_dir} with image size {args.image_size}...")
  dataset = get_dataset(
      data_dir=args.data_dir,
      image_size=args.image_size,
  )
  num_classes = len(dataset.classes)
  logger.info(f"Dataset loaded. Number of classes: {num_classes}")

  model = ViT(
    num_classes=num_classes,
    pretrained=args.use_pretrained,
    model_name=args.model_name
    ).to("cuda")
  
  optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
  criterion = torch.nn.CrossEntropyLoss()

  dataset_size = len(dataset)
  val_size = dataset_size // 10
  train_size = dataset_size - val_size
  num_training_steps = (train_size // args.batch_size) * args.num_epochs

  scheduler = None
  scheduler_per_epoch = True

  if args.scheduler_type == 'linear':
      scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=num_training_steps)
      scheduler_per_epoch = False
      logger.info(f"Using linear scheduler with {args.warmup_steps} warmup steps and {num_training_steps} total steps.")
  elif args.scheduler_type == 'plateau':
      scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=args.plateau_factor, patience=args.plateau_patience, verbose=True)
      logger.info(f"Using ReduceLROnPlateau scheduler with factor={args.plateau_factor}, patience={args.plateau_patience}.")
  else:
      logger.info("Not using a learning rate scheduler.")

  logger.info("Starting model training...")
  train_model(model, dataset, optimizer, scheduler, criterion,
              num_epochs=args.num_epochs,
              batch_size=args.batch_size,
              max_tolerant_epoch=max_tolerant_epoch_val,
              save_path=save_path_template,
              logger=logger,
              skip_optimizer_load=args.skip_optimizer_load,
              scheduler_per_epoch=scheduler_per_epoch
              )

  logger.info("Training finished.")
