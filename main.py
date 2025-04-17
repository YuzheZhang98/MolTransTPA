import torch
import wandb
import os
import logging
from torch.utils.data import DataLoader
from transformers import TrainingArguments, Trainer
from transformers.trainer_utils import get_last_checkpoint
from sklearn.model_selection import train_test_split
from dataset import TPADataset
from mol_former_tpa import MolFormerTPA
from constants import (
    BATCH_SIZE, 
    LEARNING_RATE, 
    FROZEN_NUM_EPOCHS, 
    FT_NUM_EPOCHS, 
    DEVICE, 
    USE_MIXED_PRECISION
)
import json

# Define additional constants with default values
OUTPUT_DIR = '/home/zhang2539/former_results'
LOGGING_DIR = './logs'
LOGGING_STEPS = 10
EVAL_STEPS = 500
SAVE_STEPS = 500
RANDOM_SEED = 42

# Set up logging
try:
    os.makedirs(LOGGING_DIR, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(LOGGING_DIR, "training.log")),
            logging.StreamHandler()
        ]
    )
except Exception:
    # Fallback to basic logging if file cannot be created
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
logger = logging.getLogger(__name__)


def load_data(json_file):
    """Load data from a JSON file"""
    try:
        with open(json_file, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        logger.error(f"File {json_file} not found.")
        return None
    except json.JSONDecodeError:
        logger.error(f"Error decoding JSON from {json_file}.")
        return None


def load_model_dict(file_path):
    """Load a model from a file"""
    try:
        return torch.load(file_path, map_location=DEVICE)
    except (FileNotFoundError, RuntimeError) as e:
        logger.error(f"Error loading model from {file_path}: {e}")
        return None


def compute_loss_func(outputs, labels, num_items_in_batch):
    """Compute the MSE loss function for the model"""
    predictions = outputs["predictions"]
    return torch.mean((predictions - labels)**2)


def get_example_data():
    """Return example data for testing when the real dataset is not available"""
    logger.info("Using example data for testing")
    return [
        {
            "smiles": "CCN(CC)c1ccc2c(c1)O[B-](c1ccccc1)(c1ccccc1)[N+](c1ccccc1)=C2",
            "ET(30)": 37.4,
            "dielectic constant": 7.6,
            "dipole moment": 1.75,
            "wavelength": 790,
            "TPACS": 41,
            "TPACS_log": 1.6127838567197355
        },
        {
            "smiles": "CCN(CC)c1ccc2cc(N(CC)CC)ccc2c1",
            "ET(30)": 45.6,
            "dielectic constant": 32.7,
            "dipole moment": 1.84,
            "wavelength": 800,
            "TPACS": 75,
            "TPACS_log": 1.8750612633917
        },
        {
            "smiles": "CCN(CC)c1ccc2c(c1)SC1=CC=C(N(CC)CC)C=C21",
            "ET(30)": 40.2,
            "dielectic constant": 20.4,
            "dipole moment": 1.92,
            "wavelength": 810,
            "TPACS": 110,
            "TPACS_log": 2.0413926851582249
        }
    ]


def train_model(model, train_loader, val_loader, output_dir, num_epochs, learning_rate):
    """Train the MolFormerTPA model using transformer.Trainer"""
    # Create output directory if it does not exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=train_loader.batch_size,
        per_device_eval_batch_size=val_loader.batch_size,
        eval_strategy="steps",
        eval_steps=EVAL_STEPS,
        logging_steps=LOGGING_STEPS,
        save_steps=SAVE_STEPS,
        report_to="wandb",
        logging_dir=LOGGING_DIR,
        load_best_model_at_end=True,
        fp16=USE_MIXED_PRECISION,
        save_safetensors=False,
        learning_rate=learning_rate,
        seed=RANDOM_SEED,
        dataloader_num_workers=0,
        gradient_accumulation_steps=1,
        warmup_steps=0,
        weight_decay=0.0
    )
    
    # Create the Trainer instance
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_loader.dataset,
        eval_dataset=val_loader.dataset,
        compute_loss_func=compute_loss_func,
    )
    
    # Check for existing checkpoints
    last_checkpoint = get_last_checkpoint(output_dir)
    if last_checkpoint:
        logger.info(f"Found checkpoint: {last_checkpoint}")
        logger.info("Starting fresh training instead of resuming")
    
    trainer.train()
    
    # Evaluate on the validation set
    eval_results = trainer.evaluate()
    logger.info(f"Evaluation results: {eval_results}")
    
    return output_dir


def main():
    """Main function to train and evaluate the model"""
    # Set seed for reproducibility
    torch.manual_seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)
    
    # Create required directories
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load data from JSON file
    data_file = "TPA_cleaned_data.json"
    all_data = load_data(data_file) or get_example_data()
    
    logger.info(f"Loaded {len(all_data)} data points")

    # Split data into training and validation sets
    train_data, val_data = train_test_split(all_data, test_size=0.2, random_state=RANDOM_SEED)
    logger.info(f"Training set: {len(train_data)} samples, Validation set: {len(val_data)} samples")

    # Create datasets
    train_dataset = TPADataset(train_data, is_train=True)
    val_dataset = TPADataset(val_data, is_train=False)

    # Set the scaler for validation dataset from training dataset
    val_dataset.set_scaler(train_dataset.scaler)

    # Create data loaders
    pin_memory = torch.cuda.is_available()
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=0,
        pin_memory=pin_memory
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=BATCH_SIZE,
        num_workers=0,
        pin_memory=pin_memory
    )
    
    # Initialize model with condition dimension (wavelength + 3 solvent properties)
    condition_dim = 4  # wavelength, ET(30), dielectric constant, dipole moment
    model = MolFormerTPA(condition_dim=condition_dim, use_mixed_precision=USE_MIXED_PRECISION).to(DEVICE)

    # Check if CUDA is available
    if torch.cuda.is_available():
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        logger.info("CUDA not available, using CPU")

    # Initialize wandb
    wandb.init(
        project="my-transformer-training",
        reinit=True
    )

    # Phase 1: Training with frozen MolFormer
    logger.info("Starting Phase 1: Training with frozen MolFormer")
    model.change_molformer(train_molformer=False)
    phase1_output = os.path.join(OUTPUT_DIR, "phase1")
    train_model(
        model, 
        train_loader, 
        val_loader, 
        output_dir=phase1_output,
        learning_rate=LEARNING_RATE, 
        num_epochs=FROZEN_NUM_EPOCHS
    )

    # Load the best model from Phase 1
    model_checkpoint = get_last_checkpoint(phase1_output)
    if model_checkpoint:
        trained_model_path = os.path.join(model_checkpoint, "pytorch_model.bin")
        model_dict = load_model_dict(trained_model_path)
        if model_dict:
            model.load_state_dict(model_dict)
            logger.info(f"Loaded best model from Phase 1: {trained_model_path}")
        else:
            logger.warning("Could not load best model from Phase 1, continuing with current model")
    else:
        logger.warning("No checkpoint found after Phase 1")

    # Phase 2: Fine-tuning with unfrozen MolFormer
    logger.info("Starting Phase 2: Fine-tuning with unfrozen MolFormer")
    model.change_molformer(train_molformer=True)
    phase2_output = os.path.join(OUTPUT_DIR, "phase2")
    train_model(
        model, 
        train_loader, 
        val_loader, 
        output_dir=phase2_output,
        learning_rate=LEARNING_RATE/10, 
        num_epochs=FT_NUM_EPOCHS
    )

    # Save the final model
    final_model_path = os.path.join(OUTPUT_DIR, "final_model.pt")
    torch.save(model.state_dict(), final_model_path)
    logger.info(f"Final model saved to {final_model_path}")

    # Finish the wandb run
    wandb.finish()

    logger.info("Model training complete")


if __name__ == "__main__":
    main()