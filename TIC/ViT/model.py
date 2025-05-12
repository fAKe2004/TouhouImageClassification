import transformers
from transformers import ViTForImageClassification, ViTConfig
from TIC.utils.parameter import IMAGE_SIZE, VIT_IMAGE_SIZE
import torch.nn as nn

def ViT(num_classes: int, pretrained: bool = True, model_name: str = None) -> ViTForImageClassification:
  """
  Loads a Vision Transformer (ViT) model for image classification.

  Args:
      num_classes (int): The number of output classes.
      pretrained (bool): Whether to load pretrained weights from 'google/vit-base-patch16-224-in21k'.
                         Defaults to True.

  Returns:
      ViTForImageClassification: The ViT model instance.
  """
  if model_name is None:
    # model_name = 'google/vit-base-patch16-224-in21k'
    model_name = 'google/vit-large-patch16-224-in21k'

  if pretrained:
    # Load pretrained model, adjusting the classifier head for the new number of classes
    # and allowing resizing of position embeddings if IMAGE_SIZE differs from 224.
    model = ViTForImageClassification.from_pretrained(
        model_name,
        num_labels=num_classes,
        ignore_mismatched_sizes=True # Allows adapting the final layer and position embeddings
    )
    # Ensure the config reflects the potentially new image size if different from pretrained default (224)
    if model.config.image_size != VIT_IMAGE_SIZE[0]:
      raise ValueError(
          f"Pretrained model's image size {model.config.image_size} does not match "
          f"the specified image size {VIT_IMAGE_SIZE[0]}."
      )
  else:
    # Initialize a new model from scratch with the specified configuration
    config = ViTConfig.from_pretrained(
      model_name,
      num_labels=num_classes,
    )
    model = ViTForImageClassification(config)

  return model
