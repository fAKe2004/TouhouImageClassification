from .ntrain import train_main

if __name__ == '__main__':

    PRETRAINED = True
    MODEL_NAME = 'google/vit-large-patch16-224'
    LR = 1e-5
    WEIGHT_DECAY = 0.01
    FULL_FINETUNE = True
    BATCH_SIZE = 8
    NUM_WORKERS = 4
    TRAIN_SPLIT = 0.8
    TRAIN_ID = "nViT_unfiltered"
    DATA_DIR = "data"
    MAX_EPOCHS = 20
    ENABLE_MIX_UP = True
    ENABLE_AUGMENTATION = True

    train_main(
        PRETRAINED=PRETRAINED,
        MODEL_NAME=MODEL_NAME,
        LR=LR,
        WEIGHT_DECAY=WEIGHT_DECAY,
        FULL_FINETUNE=FULL_FINETUNE,
        BATCH_SIZE=BATCH_SIZE,
        NUM_WORKERS=NUM_WORKERS,
        TRAIN_SPLIT=TRAIN_SPLIT,
        TRAIN_ID = TRAIN_ID,
        DATA_DIR=DATA_DIR,
        MAX_EPOCHS=MAX_EPOCHS,
        ENABLE_MIX_UP=ENABLE_MIX_UP,
        ENABLE_AUGMENTATION=ENABLE_AUGMENTATION,
        ONLY_GREY_AUGMENTATION=True,
    )
