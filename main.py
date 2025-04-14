import torch
import wandb
# import torch.nn as nn
# import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import TrainingArguments, Trainer
# import numpy as np
# import pandas as pd
import json
# from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
# from torch.amp import autocast, GradScaler  # For mixed precision training
from dataset import TPADataset
from mol_former_tpa import MolFormerTPA

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

    # Train model
    train_model(model, train_loader, val_loader)

    print("Model training complete and saved to ./tpa_model")

# Constants
MAX_SEQ_LENGTH = 512  # MolFormer supports longer sequences
BATCH_SIZE = 16       # Reduced due to larger model size
LEARNING_RATE = 5e-5  # Lower learning rate for fine-tuning
NUM_EPOCHS = 5
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
USE_MIXED_PRECISION = torch.cuda.is_available()  # Use mixed precision if GPU is available

# MolFormer model path
MOLFORMER_PATH = 'ibm-research/MoLFormer-XL-both-10pct'

# Solvent properties
SOLVENT_FEATURES = ['ET(30)', 'dielectic constant', 'dipole moment']

# class MolFormerTrainer(Trainer):
def compute_loss_func(outputs, labels, num_items_in_batch):
    predictions = outputs["predictions"]
    loss = torch.mean((predictions - labels)**2)  # 回归任务的MSE
    return loss

def train_model(model, train_loader, val_loader, num_epochs=NUM_EPOCHS):
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
    # Initialize wandb
    wandb.init(project="my-transformer-training", reinit=True)
    
    # Set up training arguments. Here, we use the batch size from the dataloaders.
    training_args = TrainingArguments(
        output_dir='./results',                # where to save model outputs
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
    
    # Start training.
    trainer.train()
    
    # Optionally, evaluate on the validation set.
    eval_results = trainer.evaluate()
    print("Evaluation results:", eval_results)
    
    # Finish the wandb run.
    wandb.finish()
    return model

# def predict_tpa(model, smiles, wavelength, solvent_properties):
#     """
#     Predict TPA cross-section for a single molecule

#     Args:
#         model (MolFormerTPA): Trained model
#         smiles (str): SMILES string
#         wavelength (float): Excitation wavelength
#         solvent_properties (dict): Dictionary of solvent properties

#     Returns:
#         float: Predicted TPA cross-section
#     """
#     model.eval()
#     tokenizer = AutoTokenizer.from_pretrained(MOLFORMER_PATH, trust_remote_code=True)

#     # Process SMILES
#     encoding = tokenizer(
#         smiles,
#         return_tensors='pt',
#         max_length=MAX_SEQ_LENGTH,
#         padding='max_length',
#         truncation=True
#     )

#     input_ids = encoding['input_ids'].to(DEVICE)
#     attention_mask = encoding['attention_mask'].to(DEVICE)

#     # Process wavelength (assuming we have the scaler from training)
#     # This would normally be done using the scaler from training
#     wavelength_normalized = torch.tensor([[wavelength]]).to(DEVICE)

#     # Process solvent (assuming we have encoders from training)
#     # This would normally be done using the encoders from training
#     solvent_tensor = torch.tensor([list(solvent_properties.values())], dtype=torch.float).to(DEVICE)

#     with torch.no_grad():
#         prediction = model(
#             input_ids,
#             attention_mask,
#             wavelength_normalized.squeeze(),
#             solvent_tensor.squeeze()
#         )

#     return prediction.item()

# def save_model_for_inference(model, condition_scaler, output_dir="./tpa_model"):
#     """
#     Save the trained model for later inference

#     Args:
#         model (MolFormerTPA): Trained model
#         condition_scaler (StandardScaler): Scaler for conditions
#         output_dir (str): Directory to save the model
#     """
#     import os
#     import json
#     import pickle

#     # Create directory if it doesn't exist
#     os.makedirs(output_dir, exist_ok=True)

#     # Save model weights
#     torch.save(model.state_dict(), os.path.join(output_dir, "model_weights.pt"))

#     # Save tokenizer
#     tokenizer = AutoTokenizer.from_pretrained(MOLFORMER_PATH, trust_remote_code=True)
#     tokenizer.save_pretrained(output_dir)

#     # Save condition scaler
#     with open(os.path.join(output_dir, "condition_scaler.pkl"), "wb") as f:
#         pickle.dump(condition_scaler, f)

#     # Save configuration (for recreating the model)
#     config = {
#         "max_seq_length": MAX_SEQ_LENGTH,
#         "molformer_path": MOLFORMER_PATH,
#         "condition_dim": 4,  # wavelength, ET(30), dielectric constant, dipole moment
#         "solvent_features": SOLVENT_FEATURES,
#         "use_mixed_precision": USE_MIXED_PRECISION
#     }

#     with open(os.path.join(output_dir, "config.json"), "w") as f:
#         json.dump(config, f)

#     print(f"Model saved to {output_dir}")

# def load_model_for_inference(model_dir):
#     """
#     Load a saved model for inference

#     Args:
#         model_dir (str): Directory containing the saved model

#     Returns:
#         model (MolFormerTPA): Loaded model
#         condition_scaler (StandardScaler): Scaler for conditions
#     """
#     import os
#     import json
#     import pickle

#     # Load configuration
#     with open(os.path.join(model_dir, "config.json"), "r") as f:
#         config = json.load(f)

#     # Load condition scaler
#     with open(os.path.join(model_dir, "condition_scaler.pkl"), "rb") as f:
#         condition_scaler = pickle.load(f)

#     # Initialize model
#     model = MolFormerTPA(condition_dim=config["condition_dim"])

#     # Load weights
#     model.load_state_dict(torch.load(os.path.join(model_dir, "model_weights.pt")))

#     return model.to(DEVICE), condition_scaler

# def inference_example():
#     """
#     Example of how to use the model for inference on new molecules
#     """
#     # Load the saved model
#     model_dir = "./tpa_model"
#     model, condition_scaler = load_model_for_inference(model_dir)

#     # Example molecules for prediction
#     test_molecules = [
#         {
#             "smiles": "CCN(CC)c1ccc2c(c1)O[B-](c1ccccc1)(c1ccccc1)[N+](c1ccccc1)=C2",
#             "ET(30)": 37.4,
#             "dielectic constant": 7.6,
#             "dipole moment": 1.75,
#             "wavelength": 790
#         },
#         {
#             "smiles": "c1ccc2c(c1)C(=O)c1ccccc1C2=O",  # Anthraquinone
#             "ET(30)": 42.2,
#             "dielectic constant": 4.8,
#             "dipole moment": 0.0,
#             "wavelength": 800
#         }
#     ]

#     print("\nInference on test molecules:")
#     for i, mol in enumerate(test_molecules):
#         tpa = predict_tpa(
#             model,
#             mol["smiles"],
#             mol["wavelength"],
#             mol["ET(30)"],
#             mol["dielectic constant"],
#             mol["dipole moment"],
#             condition_scaler
#         )

#         print(f"\nMolecule {i+1}:")
#         print(f"  SMILES: {mol['smiles']}")
#         print(f"  Wavelength: {mol['wavelength']} nm")
#         print(f"  Solvent properties: ET(30)={mol['ET(30)']}, Dielectric={mol['dielectic constant']}, Dipole={mol['dipole moment']}")
#         print(f"  Predicted TPA cross-section (log): {log_tpa:.4f}")
#         print(f"  Predicted TPA cross-section: {tpa:.2f} GM")

if __name__ == "__main__":
    main()
    # Uncomment to run inference example
    # inference_example()
