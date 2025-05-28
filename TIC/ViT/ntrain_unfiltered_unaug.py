from .ntrain import train_main

if __name__ == '__main__':
    train_main(
        PRETRAINED = True,
        MODEL_NAME = 'google/vit-large-patch16-224',
        LR = 1e-5,
        WEIGHT_DECAY = 0.01,
        FULL_FINETUNE = True,
        BATCH_SIZE = 16,
        NUM_WORKERS = 4,
        TRAIN_SPLIT = 0.8,
        DATA_DIR = "data",
        MAX_EPOCHS = 20,
        ENABLE_MIX_UP = False,
        ENABLE_AUGMENTATION = False,
        TRAIN_ID = "nViT_unfiltered_unaug",
        PATIENCE = -1,
    )
