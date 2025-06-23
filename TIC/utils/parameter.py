NUM_CLASSES = 120
IMAGE_SIZE = (256, 256)
VIT_IMAGE_SIZE = (224, 224)
DATA_DIR = 'data'
UNFILTERED_DATA_DIR = 'data/unfiltered'
FILTERED_DATA_DIR = 'data/data_filtered_vit_base'
CHECKPOINT_DIR = 'checkpoint'
TEST_DIR = 'data/testset'
LOG_DIR = 'log'
CACHE_DIR = 'cache'

def get_image_size(model_type: str):
  if 'vit' in model_type.lower() or 'resmoe' in model_type.lower():
    return VIT_IMAGE_SIZE
  else:
    return IMAGE_SIZE
