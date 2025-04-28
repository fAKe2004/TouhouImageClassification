import torch
from torch.utils.data import DataLoader, random_split
from TIC.ResNet.model import resnet152
from tqdm import tqdm # Import tqdm
import math

from TIC.utils.preprocess import get_dataset
import os

def train_step(model, data, optimizer, criterion):
  model.train()
  optimizer.zero_grad()
  inputs, labels = map(lambda x: x.to("cuda"), data)
  outputs = model(inputs)
  loss = criterion(outputs, labels)
  loss.backward()
  optimizer.step()
  return loss.item()

def validate_step(model, data, criterion):
  model.eval() # switch to eval mode
  with torch.no_grad(): # disable gradient calculation
    inputs, labels = map(lambda x: x.to("cuda"), data)
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    correct = (outputs.argmax(dim=1) == labels).sum().item()
  return loss.item(), correct # return validation loss

def train_model(model: torch.nn.Module, 
                dataset: torch.utils.data.Dataset, 
                optimizer: torch.optim.Optimizer, 
                criterion: torch.nn.Module, 
                batch_size: int, 
                num_epochs: int, 
                save_path: str):
  
  
  
  starting_epoch = 0
  # resume from the latest checkpoint if available
  for i in range(NUM_EPOCHS, -1, -1):
    # i is 1-indexed
    # starting_epoch is 0-indexed
    starting_epoch = i
    if i > 0 and os.path.exists(SAVE_PATH.format(epoch=i)):
      print(f"Resuming from epoch {i}")
      model.load_state_dict(torch.load(SAVE_PATH.format(epoch=i)))
      break
  
  
  # split dataloader's dataset into training and validation sets
  dataset_size = len(dataset)
  val_size = dataset_size // 10 # 1 out of 10 for validation
  train_size = dataset_size - val_size
  train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

  # Create new DataLoaders for train and validation sets
  train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
  val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

  best_val_loss = float('inf')
  epochs_no_improve = 0
  early_stop_patience = 2 # Number of epochs to wait for improvement

  def train_loop(epoch):
    
    # Training loop
    model.train() # Make sure model is in training mode
    running_train_loss = 0.0
    # Wrap train_loader with tqdm
    train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
    last_nan = None
    for i, data in enumerate(train_pbar): # Use the new train_loader
      loss = train_step(model, data, optimizer, criterion)
      running_train_loss += loss
      # Update tqdm description with current average loss
      if math.isnan(running_train_loss):
        running_train_loss = 0
        last_nan = i
      train_pbar.set_postfix({'loss': running_train_loss / (i + 1 - (last_nan if last_nan is not None else 0)), 'ever_nan': last_nan is not None})
    avg_train_loss = running_train_loss / len(train_loader)
    return avg_train_loss

  def validate_loop(epoch):
    
    # Validation loop
    running_val_loss = 0.0
    correct_sample = 0
    total_sample = 0
    # Wrap val_loader with tqdm
    val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]")
    for i, data in enumerate(val_pbar): # Use the new val_loader
      val_loss, correct = validate_step(model, data, criterion)
      running_val_loss += val_loss
      correct_sample += correct
      total_sample += len(data[1])
      
      # Update tqdm description with current average loss
      val_pbar.set_postfix({'val_loss': running_val_loss / (i + 1)})
    avg_val_loss = running_val_loss / len(val_loader)
    return avg_val_loss, correct_sample / total_sample * 100

  print("Validate before training:")
  avg_val_loss, accuracy = validate_loop(starting_epoch - 1)
  tqdm.write(f'Epoch [Before Training], Validation Loss: {avg_val_loss:.4f}, Accuracy: {accuracy}%)')
  

  for epoch in range(starting_epoch, num_epochs):
    
    avg_train_loss = train_loop(epoch)
    
    avg_val_loss, accuracy = validate_loop(epoch)
    
    # Save the model checkpoint
    torch.save(model.state_dict(), save_path.format(epoch=epoch))

    tqdm.write(f'Epoch [{epoch+1}/{num_epochs}], Training Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}, Accuracy: {accuracy}%)')

    # Early stopping check
    if avg_val_loss < best_val_loss:
      best_val_loss = avg_val_loss
      epochs_no_improve = 0
    else:
      epochs_no_improve += 1

    if epochs_no_improve >= early_stop_patience:
      print(f'Early stopping triggered after {epoch+1} epochs.')
      break # exit training loop

if __name__ == '__main__':
  # Parameters
  DATA_PATH = 'data'
  BATCH_SIZE = 15
  IMG_SIZE = (512, 512)
  NUM_EPOCHS = 10
  SAVE_PATH = 'checkpoint/ResNet_model_{epoch}.pth' # Consider adding epoch number format

  torch.set_default_dtype(torch.float16)

  # Initialize model, optimizer, and loss function
  model = resnet152().to("cuda")
  optimizer = torch.optim.SGD(model.parameters(), lr=0.005)
  criterion = torch.nn.CrossEntropyLoss()

    
  # Get DataLoader (this loader will be split inside train_model)
  dataset = get_dataset( 
      data_dir=DATA_PATH,
      img_size=IMG_SIZE,
  )

  # Train the model
  train_model(model, dataset, optimizer, criterion,
              num_epochs=NUM_EPOCHS, 
              batch_size=BATCH_SIZE,
              save_path=SAVE_PATH)