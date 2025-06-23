"""
nohup python -m TIC.ViT.train > train.out 2>&1 &
"""

import torch
from torch.utils.data import DataLoader, random_split
from torch.optim import AdamW

from tqdm import tqdm
import math
import logging
import os
import sys
from typing import Optional
from transformers import get_linear_schedule_with_warmup # Use AdamW and a suitable scheduler for Transformers

from TIC.utils.preprocess import get_dataset
from TIC.utils.parameter import *
from TIC.ViT.model import ViT # Import the ViT model

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

def train_step(model, data, optimizer, criterion, scaler, scheduler=None): # Add scheduler for potential step inside
  model.train()
  optimizer.zero_grad()
  with torch.autocast(device_type='cuda', dtype=torch.float16):
    inputs, labels = map(lambda x: x.to("cuda"), data)
    outputs = model(inputs)
    logits = outputs.logits # ViT output includes logits
    loss = criterion(logits, labels)
  scaler.scale(loss).backward()
  scaler.step(optimizer)
  scaler.update()
  if scheduler: # Step the scheduler if it's per-step (like linear warmup)
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
                scheduler_per_epoch: bool = True # Flag to indicate if scheduler steps per epoch or per step
                ):

  latest_epoch_found = 0
  latest_scheduler_state = None
  for i in range(num_epochs, 0, -1): # Check from num_epochs down to 1
      potential_path = save_path.format(epoch=i)
      if os.path.exists(potential_path):
          latest_epoch_found = i
          break

  if latest_epoch_found > 0:
      logger.info(f"Resuming from epoch {latest_epoch_found}")
      # Load checkpoint with map_location to handle potential device mismatch
      ckpt = torch.load(save_path.format(epoch=latest_epoch_found), map_location='cuda')
      if isinstance(ckpt, tuple) and len(ckpt) >= 2:
        model_state_dict, optimizer_state_dict = ckpt[0], ckpt[1]
        # Check if scheduler state is saved (optional, added in newer saves)
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
          # If skipping optimizer but using per-epoch scheduler, advance scheduler manually
          logger.info(f"Skipping optimizer load, manually advancing scheduler to epoch {latest_epoch_found}")
          # Advance the scheduler to the state it would be at the beginning of the resumed epoch
          # Note: This assumes scheduler.step() was called *after* the epoch completed in the previous run.
          for _ in range(latest_epoch_found):
              scheduler.step()

      else: # Handle older checkpoints or different formats
        model.load_state_dict(ckpt)
        logger.warning("Loaded checkpoint only contains model state_dict. Optimizer and scheduler state not loaded.")
      start_epoch = latest_epoch_found
  else:
      logger.info("Starting training from scratch.")
      start_epoch = 0


  dataset_size = len(dataset)
  val_size = dataset_size // 10
  train_size = dataset_size - val_size
  torch.manual_seed(0) # ensure split consistency across runs
  # train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
  train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

  train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=8)
  val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=8)

  avg_val_loss_timeline = []

  scaler = torch.GradScaler()

  def train_loop(epoch):
    if scheduler and scheduler_per_epoch:
      logger.info(f"LR for epoch {epoch + 1}: {scheduler.get_last_lr()[0]:.6e}")
    elif scheduler and not scheduler_per_epoch:
        # For per-step schedulers, LR changes during the epoch
        pass # Logging LR might be too verbose here

    model.train()
    running_train_loss = 0.0
    last_nan_iter = None
    train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]", file=sys.stdout)
    for i, data in enumerate(train_pbar):
      # Pass scheduler if it steps per iteration
      current_scheduler = scheduler if not scheduler_per_epoch else None
      loss = train_step(model, data, optimizer, criterion, scaler, current_scheduler)

      if math.isnan(loss):
        # Attempt to recover or log the issue
        logger.warning(f"NaN loss detected at training step {i} in epoch {epoch+1}. Replacing with avg loss.")
        # Avoid division by zero if it happens on the first step
        loss = running_train_loss / (i + 1) if i > 0 else 0.0
        last_nan_iter = i
        # Consider stopping or reducing LR if NaNs persist

      running_train_loss += loss
      current_avg_loss = running_train_loss / (i + 1)

      postfix_dict = {'loss': f'{current_avg_loss:.4f}'}
      if last_nan_iter is not None:
          postfix_dict['last_nan_iter'] = last_nan_iter
      # Optionally show current LR for per-step schedulers
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
      total_sample += len(data[1]) # data[1] should be labels
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
    avg_val_loss, accuracy = validate_loop(start_epoch - 1) # Validate the state *at the end* of the loaded epoch
    logger.info(f'Epoch [{start_epoch}], Validation Loss: {avg_val_loss:.4f}, Accuracy: {accuracy:.2f}%')
    # Optionally load past validation losses if needed for early stopping continuity
    # avg_val_loss_timeline = load_validation_history(...)

  # Training by epoch
  for epoch in range(start_epoch, num_epochs):
    avg_train_loss = train_loop(epoch)
    avg_val_loss, accuracy = validate_loop(epoch)
    avg_val_loss_timeline.append(avg_val_loss)

    # Prepare checkpoint data
    checkpoint_data = (
        model.state_dict(),
        optimizer.state_dict()
    )
    if scheduler and scheduler_per_epoch: # Only save state for per-epoch schedulers for simplicity
        checkpoint_data += (scheduler.state_dict(),)

    current_epoch_save_path = save_path.format(epoch=epoch + 1) # Save as epoch N+1 completed
    os.makedirs(os.path.dirname(current_epoch_save_path), exist_ok=True)
    torch.save(checkpoint_data, current_epoch_save_path)

    logger.info(f"Checkpoint saved to {current_epoch_save_path}")
    logger.info(f'Epoch [{epoch+1}/{num_epochs}], Training Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}, Accuracy: {accuracy:.2f}%')

    if early_exit(avg_val_loss_timeline, max_tolerant_epoch, logger):
      break

    # Step the scheduler if it's per-epoch
    if scheduler and scheduler_per_epoch:
      scheduler.step()


if __name__ == '__main__':
  # Hyperparameters for ViT (adjust as needed)
  NUM_EPOCHS = 40

  BATCH_SIZE = 30
  LR = 1e-5 # Lower learning rate is common for fine-tuning pretrained transformers
  WEIGHT_DECAY = 0.01 # Weight decay for AdamW
  SKIP_OPTIMIZER_LOAD = False # Set to true to reset optimizer/scheduler state when resuming
  USE_PRETRAINED = True # Use pretrained ViT weights
  SCHEDULER_TYPE = 'linear' # 'linear' or 'cosine' or 'none'
  WARMUP_STEPS = 500 # Number of warmup steps for the scheduler
  
  MODEL_NAME = "google/vit-large-patch16-224-in21k"

  os.makedirs(CHECKPOINT_DIR, exist_ok=True)

  SAVE_PATH = os.path.join(CHECKPOINT_DIR, 'ViT_model_finetune_{epoch}.pth')
  MAX_TOLERANT_EPOCH = NUM_EPOCHS # disable early stopping
  LOG_NAME = 'ViT_finetune'

  logger = get_logger(LOG_NAME, LOG_DIR)

  logger.info("Starting ViT training script.")
  logger.info(f"Parameters: BATCH_SIZE={BATCH_SIZE}, IMAGE_SIZE={VIT_IMAGE_SIZE}, NUM_EPOCHS={NUM_EPOCHS}, LR={LR}, PRETRAINED={USE_PRETRAINED}")

  # Use float32 for model weights, autocast handles mixed precision
  torch.set_default_dtype(torch.float32)

  logger.info(f"Loading dataset from {UNFILTERED_DATA_DIR}...")
  # Ensure get_dataset provides appropriate normalization for ViT if needed
  # Often ViT pretrained models expect normalization based on ImageNet stats
  dataset = get_dataset(
      data_dir=UNFILTERED_DATA_DIR,
      image_size=VIT_IMAGE_SIZE,
  )
  num_classes = len(dataset.classes)
  logger.info(f"Dataset loaded. Number of classes: {num_classes}")

  model = ViT(
    num_classes=num_classes,
    pretrained=USE_PRETRAINED,
    model_name=MODEL_NAME
    ).to("cuda")
  optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
  criterion = torch.nn.CrossEntropyLoss()

  # Calculate total steps for scheduler outside train_model if needed here
  # Example: num_training_steps = (len(dataset) * (1 - 0.1) // BATCH_SIZE) * NUM_EPOCHS # Approx train steps
  num_training_steps = (len(dataset) - len(dataset)//10) // BATCH_SIZE * NUM_EPOCHS # More precise estimate

  scheduler = None
  scheduler_per_epoch = False # Default to per-step scheduler if using one
  if SCHEDULER_TYPE == 'linear':
      scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=WARMUP_STEPS, num_training_steps=num_training_steps)
      logger.info(f"Using linear scheduler with {WARMUP_STEPS} warmup steps and {num_training_steps} total steps.")
  else:
      logger.info("Not using a learning rate scheduler.")
      scheduler_per_epoch = True # No scheduler behaves like a per-epoch step (no-op)


  logger.info("Starting model training...")
  train_model(model, dataset, optimizer, scheduler, criterion,
              num_epochs=NUM_EPOCHS,
              batch_size=BATCH_SIZE,
              max_tolerant_epoch=MAX_TOLERANT_EPOCH,
              save_path=SAVE_PATH,
              logger=logger,
              skip_optimizer_load=SKIP_OPTIMIZER_LOAD,
              scheduler_per_epoch=scheduler_per_epoch
              )

  logger.info("Training finished.")
