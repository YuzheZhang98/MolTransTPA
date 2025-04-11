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
    data_file = "tpa_data.json"

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
    val_dataset.set_scaler(train_dataset.condition_scaler)

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
    trained_model = train_model(model, train_loader, val_loader)

    # Example prediction
    example_smiles = "CCN(CC)c1ccc2c(c1)O[B-](c1ccccc1)(c1ccccc1)[N+](c1ccccc1)=C2"
    example_wavelength = 790
    example_et30 = 37.4
    example_dielectric = 7.6
    example_dipole = 1.75

    log_tpa, tpa = predict_tpa(
        trained_model,
        example_smiles,
        example_wavelength,
        example_et30,
        example_dielectric,
        example_dipole,
        train_dataset.condition_scaler
    )

    print(f"Predicted TPA cross-section for test molecule:")
    print(f"  Log scale: {log_tpa:.4f}")
    print(f"  Linear scale: {tpa:.2f} GM")

    # Save the model and configuration
    save_model_for_inference(
        trained_model,
        train_dataset.condition_scaler,
        output_dir="./tpa_model"
    )

    print("Model training complete and saved to ./tpa_model")
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from transformers import AutoModel, AutoTokenizer
import pandas as pd
import json
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch.cuda.amp import autocast, GradScaler  # For mixed precision training

# Constants
MAX_SEQ_LENGTH = 512  # MolFormer supports longer sequences
BATCH_SIZE = 16       # Reduced due to larger model size
LEARNING_RATE = 5e-5  # Lower learning rate for fine-tuning
NUM_EPOCHS = 30
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
USE_MIXED_PRECISION = torch.cuda.is_available()  # Use mixed precision if GPU is available

# MolFormer model path
MOLFORMER_PATH = 'ibm/MolFormer'

# Solvent properties
SOLVENT_FEATURES = ['ET(30)', 'dielectic constant', 'dipole moment']

# Solvent properties - example features
SOLVENT_FEATURES = ['polarity', 'viscosity', 'refractive_index']
CATEGORICAL_SOLVENTS = ['water', 'methanol', 'dmso', 'acetone', 'chloroform', 'toluene']

class TPADataset(Dataset):
    def __init__(self, data_list, is_train=True):
        """
        Dataset for two-photon absorption prediction

        Args:
            data_list (list): List of dictionaries containing SMILES, solvent properties,
                              wavelength, and TPA cross-section values
            is_train (bool): Whether this is training data (for scaling)
        """
        self.tokenizer = AutoTokenizer.from_pretrained(MOLFORMER_PATH)
        self.data = data_list
        self.is_train = is_train

        # Extract features and targets
        smiles_list = [item['smiles'] for item in data_list]
        wavelengths = np.array([item['wavelength'] for item in data_list]).reshape(-1, 1)
        et30_values = np.array([item['ET(30)'] for item in data_list]).reshape(-1, 1)
        dielectric_constants = np.array([item['dielectic constant'] for item in data_list]).reshape(-1, 1)
        dipole_moments = np.array([item['dipole moment'] for item in data_list]).reshape(-1, 1)

        # Use log TPACS as target (provided in the data)
        if 'TPACS_log' in data_list[0]:
            tpa_values = np.array([item['TPACS_log'] for item in data_list])
        else:
            tpa_values = np.array([item['TPACS'] for item in data_list])
            # Log transform if using raw TPACS values and log values not provided
            tpa_values = np.log10(tpa_values)

        self.smiles_list = smiles_list
        self.tpa_values = tpa_values

        # Combine solvent properties and wavelength for the conditional encoder
        self.condition_data = np.hstack([wavelengths, et30_values, dielectric_constants, dipole_moments])

        # Initialize or fit scalers
        if is_train:
            self.condition_scaler = StandardScaler()
            self.scaled_condition_data = self.condition_scaler.fit_transform(self.condition_data)
        else:
            # Assume scaler is set later for validation/test data
            self.condition_scaler = None
            self.scaled_condition_data = self.condition_data  # Will be updated later

    def set_scaler(self, scaler):
        """Set the scaler from training data for validation/test data"""
        if not self.is_train:
            self.condition_scaler = scaler
            self.scaled_condition_data = self.condition_scaler.transform(self.condition_data)

    def __len__(self):
        return len(self.smiles_list)

    def __getitem__(self, idx):
        smiles = self.smiles_list[idx]

        # Tokenize SMILES
        encoding = self.tokenizer(
            smiles,
            return_tensors='pt',
            max_length=MAX_SEQ_LENGTH,
            padding='max_length',
            truncation=True
        )

        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()

        # Get conditions (wavelength and solvent properties)
        condition = torch.tensor(self.scaled_condition_data[idx], dtype=torch.float32)

        # Get target
        tpa = torch.tensor(self.tpa_values[idx], dtype=torch.float32)

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'condition': condition,
            'tpa': tpa
        }

class ConditionEncoder(nn.Module):
    def __init__(self, condition_dim=4, hidden_dim=64, output_dim=128):
        """
        MLP to encode wavelength and solvent properties

        Args:
            condition_dim (int): Dimension of condition input (wavelength + 3 solvent properties)
            hidden_dim (int): Hidden dimension
            output_dim (int): Output dimension
        """
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(condition_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, condition):
        return self.mlp(condition)

class TPAPredictionHead(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        """
        Prediction head for TPA cross-section

        Args:
            input_dim (int): Input dimension
            hidden_dim (int): Hidden dimension
        """
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, x):
        return self.mlp(x)

class MolFormerTPA(nn.Module):
    def __init__(self, condition_dim=4):
        """
        Model for predicting two-photon absorption using MolFormer

        Args:
            condition_dim (int): Dimension of conditions (wavelength + solvent properties)
        """
        super().__init__()

        # 1. SMILES Encoding Branch - MolFormer
        self.molformer = AutoModel.from_pretrained(MOLFORMER_PATH)
        molformer_dim = self.molformer.config.hidden_size

        # 2. Conditions Encoding Branch
        self.condition_encoder = ConditionEncoder(
            condition_dim=condition_dim,
            output_dim=128
        )

        # 3. Fusion Layer
        self.fusion = nn.Linear(molformer_dim + 128, 256)

        # 4. Prediction Head
        self.prediction_head = TPAPredictionHead(input_dim=256)

    def forward(self, input_ids, attention_mask, condition):
        # 1. SMILES Encoding
        with autocast(enabled=USE_MIXED_PRECISION):
            molformer_output = self.molformer(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True
            )

            # Use CLS token as molecular representation
            mol_repr = molformer_output.last_hidden_state[:, 0, :]

        # 2. Conditions Encoding
        cond_repr = self.condition_encoder(condition)

        # 3. Fusion
        combined = torch.cat([mol_repr, cond_repr], dim=1)
        fused = F.relu(self.fusion(combined))

        # 4. Prediction
        tpa_prediction = self.prediction_head(fused)

        return tpa_prediction.squeeze()

def train_model(model, train_loader, val_loader, num_epochs=NUM_EPOCHS):
    """
    Train the MolFormerTPA model with mixed precision

    Args:
        model (MolFormerTPA): The model
        train_loader (DataLoader): Training data loader
        val_loader (DataLoader): Validation data loader
        num_epochs (int): Number of epochs

    Returns:
        model (MolFormerTPA): Trained model
    """
    # Different learning rates for pre-trained model and new layers
    optimizer = torch.optim.AdamW([
        {'params': model.molformer.parameters(), 'lr': LEARNING_RATE / 10},  # Lower LR for pre-trained
        {'params': model.condition_encoder.parameters()},
        {'params': model.fusion.parameters()},
        {'params': model.prediction_head.parameters()}
    ], lr=LEARNING_RATE, weight_decay=0.01)

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, verbose=True
    )

    # Loss function - MSE for regression
    criterion = nn.MSELoss()

    # Initialize gradient scaler for mixed precision training
    scaler = GradScaler(enabled=USE_MIXED_PRECISION)

    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0

        for batch in train_loader:
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            condition = batch['condition'].to(DEVICE)
            tpa = batch['tpa'].to(DEVICE)

            optimizer.zero_grad()

            # Forward pass with mixed precision
            with autocast(enabled=USE_MIXED_PRECISION):
                predictions = model(input_ids, attention_mask, condition)
                loss = criterion(predictions, tpa)

            # Backward pass with gradient scaling
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validation
        model.eval()
        val_loss = 0
        val_predictions = []
        val_targets = []

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(DEVICE)
                attention_mask = batch['attention_mask'].to(DEVICE)
                condition = batch['condition'].to(DEVICE)
                tpa = batch['tpa'].to(DEVICE)

                # Mixed precision for validation too
                with autocast(enabled=USE_MIXED_PRECISION):
                    predictions = model(input_ids, attention_mask, condition)
                    loss = criterion(predictions, tpa)

                val_loss += loss.item()
                val_predictions.extend(predictions.cpu().numpy())
                val_targets.extend(tpa.cpu().numpy())

            val_loss /= len(val_loader)

            # Calculate additional metrics
            val_mae = np.mean(np.abs(np.array(val_predictions) - np.array(val_targets)))
            val_r2 = np.corrcoef(val_predictions, val_targets)[0, 1] ** 2

        print(f'Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val MAE: {val_mae:.4f} | Val R²: {val_r2:.4f}')

        # Update learning rate based on validation loss
        scheduler.step(val_loss)

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'molformer_tpa_best.pt')

    # Load best model
    model.load_state_dict(torch.load('molformer_tpa_best.pt'))
    return model

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'molformer_tpa_best.pt')

    # Load best model
    model.load_state_dict(torch.load('molformer_tpa_best.pt'))
    return model

def predict_tpa(model, smiles, wavelength, solvent_properties):
    """
    Predict TPA cross-section for a single molecule

    Args:
        model (MolFormerTPA): Trained model
        smiles (str): SMILES string
        wavelength (float): Excitation wavelength
        solvent_properties (dict): Dictionary of solvent properties

    Returns:
        float: Predicted TPA cross-section
    """
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained('ibm/MolFormer')

    # Process SMILES
    encoding = tokenizer(
        smiles,
        return_tensors='pt',
        max_length=MAX_SEQ_LENGTH,
        padding='max_length',
        truncation=True
    )

    input_ids = encoding['input_ids'].to(DEVICE)
    attention_mask = encoding['attention_mask'].to(DEVICE)

    # Process wavelength (assuming we have the scaler from training)
    # This would normally be done using the scaler from training
    wavelength_normalized = torch.tensor([[wavelength]]).to(DEVICE)

    # Process solvent (assuming we have encoders from training)
    # This would normally be done using the encoders from training
    solvent_tensor = torch.tensor([list(solvent_properties.values())], dtype=torch.float).to(DEVICE)

    with torch.no_grad():
        prediction = model(
            input_ids,
            attention_mask,
            wavelength_normalized.squeeze(),
            solvent_tensor.squeeze()
        )

    return prediction.item()

def main():
    """
    Main function to demonstrate usage
    """
    # Example data loading (replace with your actual data)
    # In a real implementation, you would load this from CSV files
    example_data = {
        'smiles': [
            'CCO',  # Ethanol
            'CC1=CC=CC=C1',  # Toluene
            'C1=CC=C(C=C1)C=O',  # Benzaldehyde
            'c1ccc2c(c1)C(=O)c3ccccc3C2=O',  # Anthraquinone
            'c1cc(cc(c1)N)N',  # m-Phenylenediamine
            'Clc1ccc2c(c1)C(=O)C3=C(C2=O)C(=CC=C3)Cl',  # 1,5-Dichloroanthraquinone
            'C1=CC=C2C(=C1)C(=O)C3=CC=CC=C3C2=O',  # Anthracene-9,10-dione
            'c1ccc2nc3ccccc3cc2c1',  # Acridine
            # Add more representative molecules for TPA study
        ],
        'wavelength': [
            800,
            850,
            900,
            750,
            780,
            820,
            860,
            880,
            # Add more wavelengths
        ],
        'solvent_type': [
            'water',
            'dmso',
            'methanol',
            'acetone',
            'chloroform',
            'toluene',
            'water',
            'dmso',
            # Add more solvent types
        ],
        'polarity': [
            78.4,  # Water
            46.7,  # DMSO
            32.6,  # Methanol
            20.7,  # Acetone
            4.8,   # Chloroform
            2.4,   # Toluene
            78.4,  # Water
            46.7,  # DMSO
            # Add more polarity values
        ],
        'viscosity': [
            0.89,  # Water
            1.99,  # DMSO
            0.54,  # Methanol
            0.31,  # Acetone
            0.56,  # Chloroform
            0.59,  # Toluene
            0.89,  # Water
            1.99,  # DMSO
            # Add more viscosity values
        ],
        'tpa': [
            15.2,   # Example TPA values in GM units (Goeppert-Mayer)
            45.7,   # These would be experimental values
            32.1,   # or calculated from high-level quantum methods
            120.5,  # Higher values for more conjugated systems
            85.3,
            210.7,
            175.2,
            95.8,
            # Add more TPA values
        ]
    }

    # Convert to DataFrame
    df = pd.DataFrame(example_data)

    # Split data
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    # Create datasets
    train_dataset = TPADataset(
        train_df['smiles'].tolist(),
        train_df['wavelength'].tolist(),
        train_df[['solvent_type', 'polarity', 'viscosity']],
        train_df['tpa'].tolist()
    )

    test_dataset = TPADataset(
        test_df['smiles'].tolist(),
        test_df['wavelength'].tolist(),
        test_df[['solvent_type', 'polarity', 'viscosity']],
        test_df['tpa'].tolist()
    )

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    # Determine solvent dimension from dataset
    sample = next(iter(train_loader))
    solvent_dim = sample['solvent'].shape[1]

    # Initialize model
    model = MolFormerTPA(solvent_dim=solvent_dim).to(DEVICE)

    # Train model
    trained_model = train_model(model, train_loader, test_loader)

    # Example prediction
    example_smiles = 'CCO'  # Ethanol
    example_wavelength = 800  # nm
    example_solvent = {
        'solvent_type_water': 1,  # One-hot encoded
        'solvent_type_methanol': 0,
        'solvent_type_dmso': 0,
        'solvent_type_acetone': 0,
        'solvent_type_chloroform': 0,
        'solvent_type_toluene': 0,
        'polarity': 78.4,
        'viscosity': 0.89
    }

    predicted_tpa = predict_tpa(trained_model, example_smiles, example_wavelength, example_solvent)
    print(f"Predicted TPA cross-section for {example_smiles} at {example_wavelength} nm: {predicted_tpa}")

def save_model_for_inference(model, condition_scaler, output_dir="./tpa_model"):
    """
    Save the trained model for later inference

    Args:
        model (MolFormerTPA): Trained model
        condition_scaler (StandardScaler): Scaler for conditions
        output_dir (str): Directory to save the model
    """
    import os
    import json
    import pickle

    # Create directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Save model weights
    torch.save(model.state_dict(), os.path.join(output_dir, "model_weights.pt"))

    # Save tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MOLFORMER_PATH)
    tokenizer.save_pretrained(output_dir)

    # Save condition scaler
    with open(os.path.join(output_dir, "condition_scaler.pkl"), "wb") as f:
        pickle.dump(condition_scaler, f)

    # Save configuration (for recreating the model)
    config = {
        "max_seq_length": MAX_SEQ_LENGTH,
        "molformer_path": MOLFORMER_PATH,
        "condition_dim": 4,  # wavelength, ET(30), dielectric constant, dipole moment
        "solvent_features": SOLVENT_FEATURES,
        "use_mixed_precision": USE_MIXED_PRECISION
    }

    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(config, f)

    print(f"Model saved to {output_dir}")

def load_model_for_inference(model_dir):
    """
    Load a saved model for inference

    Args:
        model_dir (str): Directory containing the saved model

    Returns:
        model (MolFormerTPA): Loaded model
        condition_scaler (StandardScaler): Scaler for conditions
    """
    import os
    import json
    import pickle

    # Load configuration
    with open(os.path.join(model_dir, "config.json"), "r") as f:
        config = json.load(f)

    # Load condition scaler
    with open(os.path.join(model_dir, "condition_scaler.pkl"), "rb") as f:
        condition_scaler = pickle.load(f)

    # Initialize model
    model = MolFormerTPA(condition_dim=config["condition_dim"])

    # Load weights
    model.load_state_dict(torch.load(os.path.join(model_dir, "model_weights.pt")))

    return model.to(DEVICE), condition_scaler

def inference_example():
    """
    Example of how to use the model for inference on new molecules
    """
    # Load the saved model
    model_dir = "./tpa_model"
    model, condition_scaler = load_model_for_inference(model_dir)

    # Example molecules for prediction
    test_molecules = [
        {
            "smiles": "CCN(CC)c1ccc2c(c1)O[B-](c1ccccc1)(c1ccccc1)[N+](c1ccccc1)=C2",
            "ET(30)": 37.4,
            "dielectic constant": 7.6,
            "dipole moment": 1.75,
            "wavelength": 790
        },
        {
            "smiles": "c1ccc2c(c1)C(=O)c1ccccc1C2=O",  # Anthraquinone
            "ET(30)": 42.2,
            "dielectic constant": 4.8,
            "dipole moment": 0.0,
            "wavelength": 800
        }
    ]

    print("\nInference on test molecules:")
    for i, mol in enumerate(test_molecules):
        log_tpa, tpa = predict_tpa(
            model,
            mol["smiles"],
            mol["wavelength"],
            mol["ET(30)"],
            mol["dielectic constant"],
            mol["dipole moment"],
            condition_scaler
        )

        print(f"\nMolecule {i+1}:")
        print(f"  SMILES: {mol['smiles']}")
        print(f"  Wavelength: {mol['wavelength']} nm")
        print(f"  Solvent properties: ET(30)={mol['ET(30)']}, Dielectric={mol['dielectic constant']}, Dipole={mol['dipole moment']}")
        print(f"  Predicted TPA cross-section (log): {log_tpa:.4f}")
        print(f"  Predicted TPA cross-section: {tpa:.2f} GM")

if __name__ == "__main__":
    main()
    # Uncomment to run inference example
    # inference_example()
