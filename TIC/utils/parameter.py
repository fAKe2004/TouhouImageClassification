NUM_CLASSES = 120
IMAGE_SIZE = (256, 256)
VIT_IMAGE_SIZE = (224, 224)
DATA_DIR = 'data'
CHECKPOINT_DIR = 'checkpoint'
LOG_DIR = 'log'

def get_image_size(model_type: str):
  if model_type.lower() == 'vit':
    return VIT_IMAGE_SIZE
  else:
    return IMAGE_SIZE