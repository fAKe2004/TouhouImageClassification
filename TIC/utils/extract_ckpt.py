import torch

if __name__ == "__main__":
  import os
  import argparse
  parser = argparse.ArgumentParser("Extract model checkpoint from training checkpoint")
  parser.add_argument("input_path", type=str, help="Path to the input training checkpoint")
  parser.add_argument("output_path", type=str, help="Path to the output model checkpoint")
  
  args = parser.parse_args()
  
  # Load the training checkpoint
  try:
    ckpt = torch.load(args.input_path, weights_only=False)
    print(f"Loaded checkpoint from {args.input_path}")
  except FileNotFoundError:
    print(f"Checkpoint file not found: {args.input_path}")
    exit(1)
  try: 
    ckpt = ckpt["model"] 
  except Exception:
    print("No 'model' key found in the checkpoint. Attempt to interpret as Tuple")
    try:
      ckpt = ckpt[0]
    except Exception:
      exit(1)
    
  torch.save(ckpt, args.output_path)
  print(f"Extracted model checkpoint saved to {args.output_path}")

"""
python -m TIC.utils.extract_ckpt input_path output_path
"""
  