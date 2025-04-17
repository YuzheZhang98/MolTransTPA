import torch
# Constants for the TPA prediction model

# Model paths
MOLFORMER_PATH = "ibm-research/MoLFormer-XL-both-10pct"

# Device settings
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_MIXED_PRECISION = False

# Data processing settings
MAX_SEQ_LENGTH = 512  # Maximum length for SMILES tokenization

# Training hyperparameters
BATCH_SIZE = 16
LEARNING_RATE = 5e-4
FROZEN_NUM_EPOCHS = 5
FT_NUM_EPOCHS = 5

# Logging and checkpoints
CHECKPOINT_DIR = "./checkpoints"
LOG_INTERVAL = 100  # Log training metrics every N steps
