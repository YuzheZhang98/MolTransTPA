import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from transformers import RobertaModel, RobertaTokenizer
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split

# Constants
MAX_SEQ_LENGTH = 128
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
NUM_EPOCHS = 50
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Solvent properties - example features
SOLVENT_FEATURES = ['polarity', 'viscosity', 'refractive_index']
CATEGORICAL_SOLVENTS = ['water', 'methanol', 'dmso', 'acetone', 'chloroform', 'toluene']

class TPADataset(Dataset):
    def __init__(self, smiles_list, wavelengths, solvent_data, tpa_values):
        """
        Dataset for two-photon absorption prediction

        Args:
            smiles_list (list): List of SMILES strings
            wavelengths (list): List of excitation wavelengths
            solvent_data (pd.DataFrame): DataFrame with solvent properties
            tpa_values (list): Target TPA cross-section values
        """
        self.tokenizer = RobertaTokenizer.from_pretrained('seyonec/MolFormer')
        self.smiles_list = smiles_list
        self.wavelengths = wavelengths
        self.solvent_data = solvent_data
        self.tpa_values = tpa_values

        # Normalize wavelengths
        self.wavelength_scaler = StandardScaler()
        self.normalized_wavelengths = self.wavelength_scaler.fit_transform(
            np.array(wavelengths).reshape(-1, 1)
        ).flatten()

        # Process solvent features
        self.solvent_continuous_scaler = StandardScaler()
        self.solvent_categorical_encoder = OneHotEncoder(sparse=False)

        # Split solvent data into continuous and categorical
        if 'solvent_type' in self.solvent_data.columns:
            self.solvent_cat = self.solvent_categorical_encoder.fit_transform(
                self.solvent_data[['solvent_type']]
            )
            continuous_cols = [col for col in self.solvent_data.columns if col != 'solvent_type']
            if continuous_cols:
                self.solvent_cont = self.solvent_continuous_scaler.fit_transform(
                    self.solvent_data[continuous_cols]
                )
            else:
                self.solvent_cont = np.array([])
        else:
            self.solvent_cat = np.array([])
            self.solvent_cont = self.solvent_continuous_scaler.fit_transform(self.solvent_data)

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

        # Get wavelength
        wavelength = torch.tensor(self.normalized_wavelengths[idx], dtype=torch.float)

        # Get solvent features
        solvent_features = []
        if len(self.solvent_cat) > 0:
            solvent_features.append(torch.tensor(self.solvent_cat[idx], dtype=torch.float))
        if len(self.solvent_cont) > 0:
            solvent_features.append(torch.tensor(self.solvent_cont[idx], dtype=torch.float))

        solvent = torch.cat(solvent_features) if solvent_features else torch.tensor([])

        # Get target
        tpa = torch.tensor(self.tpa_values[idx], dtype=torch.float)

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'wavelength': wavelength,
            'solvent': solvent,
            'tpa': tpa
        }

class ConditionEncoder(nn.Module):
    def __init__(self, wavelength_dim=1, solvent_dim=10, hidden_dim=64, output_dim=128):
        """
        MLP to encode wavelength and solvent properties

        Args:
            wavelength_dim (int): Dimension of wavelength input
            solvent_dim (int): Dimension of solvent features
            hidden_dim (int): Hidden dimension
            output_dim (int): Output dimension
        """
        super().__init__()
        self.input_dim = wavelength_dim + solvent_dim

        self.mlp = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, wavelength, solvent):
        # Concatenate wavelength and solvent features
        x = torch.cat([wavelength.unsqueeze(1), solvent], dim=1)
        return self.mlp(x)

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
    def __init__(self, solvent_dim=10):
        """
        Model for predicting two-photon absorption using MolFormer

        Args:
            solvent_dim (int): Dimension of solvent features
        """
        super().__init__()

        # 1. SMILES Encoding Branch - MolFormer
        self.molformer = RobertaModel.from_pretrained('seyonec/MolFormer')
        molformer_dim = self.molformer.config.hidden_size

        # 2. Conditions Encoding Branch
        self.condition_encoder = ConditionEncoder(
            wavelength_dim=1,
            solvent_dim=solvent_dim,
            output_dim=128
        )

        # 3. Fusion Layer
        self.fusion = nn.Linear(molformer_dim + 128, 256)

        # 4. Prediction Head
        self.prediction_head = TPAPredictionHead(input_dim=256)

    def forward(self, input_ids, attention_mask, wavelength, solvent):
        # 1. SMILES Encoding
        molformer_output = self.molformer(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        # Use CLS token as molecular representation
        mol_repr = molformer_output.last_hidden_state[:, 0, :]

        # 2. Conditions Encoding
        cond_repr = self.condition_encoder(wavelength, solvent)

        # 3. Fusion
        combined = torch.cat([mol_repr, cond_repr], dim=1)
        fused = F.relu(self.fusion(combined))

        # 4. Prediction
        tpa_prediction = self.prediction_head(fused)

        return tpa_prediction.squeeze()

def train_model(model, train_loader, val_loader, num_epochs=NUM_EPOCHS):
    """
    Train the MolFormerTPA model

    Args:
        model (MolFormerTPA): The model
        train_loader (DataLoader): Training data loader
        val_loader (DataLoader): Validation data loader
        num_epochs (int): Number of epochs

    Returns:
        model (MolFormerTPA): Trained model
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()

    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0

        for batch in train_loader:
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            wavelength = batch['wavelength'].to(DEVICE)
            solvent = batch['solvent'].to(DEVICE)
            tpa = batch['tpa'].to(DEVICE)

            optimizer.zero_grad()

            predictions = model(input_ids, attention_mask, wavelength, solvent)
            loss = criterion(predictions, tpa)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validation
        model.eval()
        val_loss = 0

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(DEVICE)
                attention_mask = batch['attention_mask'].to(DEVICE)
                wavelength = batch['wavelength'].to(DEVICE)
                solvent = batch['solvent'].to(DEVICE)
                tpa = batch['tpa'].to(DEVICE)

                predictions = model(input_ids, attention_mask, wavelength, solvent)
                loss = criterion(predictions, tpa)

                val_loss += loss.item()

            val_loss /= len(val_loader)

        print(f'Epoch {epoch+1}/{num_epochs} | Training Loss: {train_loss:.4f} | Validation Loss: {val_loss:.4f}')

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
    tokenizer = RobertaTokenizer.from_pretrained('seyonec/MolFormer')

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
    # In a real implementation, you would load this from files
    example_data = {
        'smiles': [
            'CCO',
            'CC1=CC=CC=C1',
            'C1=CC=C(C=C1)C=O',
            # Add more molecules
        ],
        'wavelength': [
            800,
            850,
            900,
            # Add more wavelengths
        ],
        'solvent_type': [
            'water',
            'dmso',
            'methanol',
            # Add more solvent types
        ],
        'polarity': [
            78.4,
            46.7,
            32.6,
            # Add more polarity values
        ],
        'viscosity': [
            0.89,
            1.99,
            0.54,
            # Add more viscosity values
        ],
        'tpa': [
            15.2,
            45.7,
            32.1,
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

if __name__ == "__main__":
    main()
