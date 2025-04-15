import torch
import wandb
# import torch.nn as nn
# import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import TrainingArguments, Trainer, AutoModel
# import numpy as np
# import pandas as pd
import json
# from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
# from torch.amp import autocast, GradScaler  # For mixed precision training
from dataset import TPADataset
from mol_former_tpa import MolFormerTPA
from constants import BATCH_SIZE, LEARNING_RATE, NUM_EPOCHS, DEVICE, USE_MIXED_PRECISION
from transformers.trainer_utils import get_last_checkpoint

def load_data(json_file):
    """
    Load data from a JSON file

    Args:
        json_file (str): Path to JSON file

    Returns:
        list: List of data entries
    """
    with open(json_file, 'r') as f:
        data = json.load(f)
    return data

def load_model_dict(file_path):
    with open(file_path, 'rb') as f:
        model = torch.load(f, map_location=DEVICE)

    return model

def main():
    """
    Main function to train and evaluate the model
    """
    # Load data from JSON file
    # Replace with your actual JSON file path
    data_file = "TPA_cleaned_data.json"

    try:
        all_data = load_data(data_file)
        print(f"Loaded {len(all_data)} data points from {data_file}")
    except FileNotFoundError:
        print(f"File {data_file} not found. Using example data.")
        # Example data in your format
        all_data = [
            {
                "smiles": "CCN(CC)c1ccc2c(c1)O[B-](c1ccccc1)(c1ccccc1)[N+](c1ccccc1)=C2",
                "ET(30)": 37.4,
                "dielectic constant": 7.6,
                "dipole moment": 1.75,
                "wavelength": 790,
                "TPACS": 41,
                "TPACS_log": 1.6127838567197355
            },
            # Add more dummy examples here
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

    # Split data into training and validation sets
    train_data, val_data = train_test_split(all_data, test_size=0.2, random_state=42)

    # Create datasets
    train_dataset = TPADataset(train_data, is_train=True)
    val_dataset = TPADataset(val_data, is_train=False)

    # Set the scaler for validation dataset from training dataset
    val_dataset.set_scaler(train_dataset.scaler)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    # Initialize model with correct condition dimension (wavelength + 3 solvent properties)
    condition_dim = 4  # wavelength, ET(30), dielectric constant, dipole moment
    model = MolFormerTPA(condition_dim=condition_dim).to(DEVICE)

    # Enable mixed precision training if available
    if USE_MIXED_PRECISION:
        print("Using mixed precision training")

    # Initialize wandb
    wandb.init(project="my-transformer-training", reinit=True)

    # Start training without tuning MolFormer with high training rate.
    model.change_molformer(train_molformer=False)
    train_model(model, train_loader, val_loader, learning_rate=LEARNING_RATE, num_epochs=NUM_EPOCHS)

    trained_model_path = get_last_checkpoint("/internfs/duzhilin/former_results") + "/pytorch_model.bin"
    model.load_state_dict(load_model_dict(trained_model_path))

    # Train the model with tuning MolFormer with low training rate.
    model.change_molformer(train_molformer=True)
    train_model(model, train_loader, val_loader, learning_rate=LEARNING_RATE / 10, num_epochs=10)

    # Finish the wandb run.
    wandb.finish()

    print("Model training complete and saved to ./tpa_model")

def compute_loss_func(outputs, labels, num_items_in_batch):
    predictions = outputs["predictions"]
    loss = torch.mean((predictions - labels)**2)  # 回归任务的MSE
    return loss

def train_model(model, train_loader, val_loader, num_epochs=NUM_EPOCHS, learning_rate=5e-5):
    """
    Train the MolFormerTPA model with mixed precision using transformer.Trainer

    Args:
        model (MolFormerTPA): The model
        train_loader (DataLoader): Training data loader
        val_loader (DataLoader): Validation data loader
        num_epochs (int): Number of epochs

    Returns:
        model (MolFormerTPA): Trained model
    """
    
    # Set up training arguments. Here, we use the batch size from the dataloaders.
    training_args = TrainingArguments(
        output_dir='/internfs/duzhilin/former_results',                # where to save model outputs
        num_train_epochs=num_epochs,           # total number of training epochs
        per_device_train_batch_size=train_loader.batch_size,
        per_device_eval_batch_size=val_loader.batch_size,
        eval_strategy="steps",                 # evaluation is done (and logged) every eval_steps
        eval_steps=500,                        # number of training steps between evaluations
        logging_steps=10,                      # interval for logging
        save_steps=500,                        # save checkpoint every save_steps
        report_to="wandb",                     # enable logging to wandb
        logging_dir='./logs',
        load_best_model_at_end=True,           # optionally load best model at end of training
        fp16=True,                             # enable mixed precision training
        save_safetensors=False,
        learning_rate=learning_rate
    )
    
    # Create the Trainer instance.
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_loader.dataset,
        eval_dataset=val_loader.dataset,
        compute_loss_func=compute_loss_func,
        # compute_metrics=compute_metrics,       # function to compute metrics
    )
    
    trainer.train()
    
    # Optionally, evaluate on the validation set.
    eval_results = trainer.evaluate()
    print("Evaluation results:", eval_results)
    
    return model

if __name__ == "__main__":
    main()
    # Uncomment to run inference example
    # inference_example()
