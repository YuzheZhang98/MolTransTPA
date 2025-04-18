import torch
import os
# Constants for the TPA prediction model

# Model paths
MOLFORMER_PATH = "ibm-research/MoLFormer-XL-both-10pct"

# Directory configuration
OUTPUT_DIR = './results'
LOGGING_DIR = './logs'

# Training hyperparameters
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
FROZEN_NUM_EPOCHS = 20
FT_NUM_EPOCHS = 10
RANDOM_SEED = 42

# Hardware configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_MIXED_PRECISION = False

# Model configuration
HIDDEN_DIM = 256
CONDITION_DIM = 4  # wavelength, ET(30), dielectric constant, dipole moment

# Augmentation
AUGMENTATION_COUNT = 2  # Number of augmented SMILES per original SMILES

# Validation
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Set environment variable to avoid tokenizers warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"