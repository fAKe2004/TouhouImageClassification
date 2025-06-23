from .ntrain import train_main
from TIC.utils.parameter import FILTERED_DATA_DIR

if __name__ == '__main__':
    train_main(
        PRETRAINED = True,
        MODEL_NAME = 'google/vit-large-patch16-224',
        LR = 1e-5,
        WEIGHT_DECAY = 0.01,
        FULL_FINETUNE = True,
        BATCH_SIZE = 8,
        NUM_WORKERS = 4,
        TRAIN_SPLIT = 0.8,
        TRAIN_ID = "nViT_but_div",
        DATA_DIR = FILTERED_DATA_DIR,
        MAX_EPOCHS = 20,
        ENABLE_MIX_UP = True,
        ENABLE_AUGMENTATION = True,
        ENABLE_DIVERSITY = False,
    )
