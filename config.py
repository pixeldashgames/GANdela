import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TRAIN_DIR = "./database/train"
VAL_DIR = "./database/validation"
LEARNING_RATE_GEN = 2e-4  # Initial learning rate for the generator
LEARNING_RATE_DISC = 2e-4  # Initial learning rate for the discriminator
BETA1 = 0.5
BATCH_SIZE = 16
NUM_WORKERS = 2
IMAGE_SIZE = 256
CHANNELS_IMG = 3
L1_LAMBDA = 10
LAMBDA_GP = 10
NUM_EPOCHS = 400
LOAD_MODEL = False
SAVE_MODEL = True
CHECKPOINT_DISC = "disc.pth.tar"
CHECKPOINT_GEN = "gen.pth.tar"
