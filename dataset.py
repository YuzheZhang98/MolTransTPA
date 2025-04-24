from torch.utils.data import Dataset
import numpy as np
import torch
from transformers import AutoTokenizer
from constants import MOLFORMER_PATH
import os

# Set environment variable to avoid tokenizers warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def validate_dataset(data_list):
    """Validate dataset entries to prevent NaN issues
   
    Args:
        data_list (list): List of data dictionaries
       
    Returns:
        tuple: (valid_data, report) - Filtered data and validation report
    """
    report = {"total": len(data_list), "valid": 0, "filtered": 0, "issues": {}}
    valid_data = []
   
    for item in data_list:
        issues = []
       
        # Check SMILES
        if not item.get('smiles'):
            issues.append("missing_smiles")
       
        # Check for zeros or negative values in TPACS
        if item.get('TPACS', 0) <= 0:
            issues.append("invalid_tpacs")
       
        # Check for NaN or infinite values in numerical fields
        for field in ['ET(30)', 'dielectic constant', 'dipole moment', 'wavelength']:
            if field not in item or not np.isfinite(float(item.get(field, np.nan))):
                issues.append(f"invalid_{field}")
       
        # Keep only valid entries
        if not issues:
            valid_data.append(item)
            report["valid"] += 1
        else:
            report["filtered"] += 1
            for issue in issues:
                report["issues"][issue] = report["issues"].get(issue, 0) + 1
   
    return valid_data, report


def smiles_augmentation(smiles_list, num_augmentations=2):
    """
    Perform SMILES augmentation by generating multiple valid SMILES
    for the same molecule using RDKit.
    
    Args:
        smiles_list (list): List of SMILES strings
        num_augmentations (int): Number of augmentations per molecule
        
    Returns:
        dict: Mapping of original SMILES to list of augmented SMILES
    """
    import random
    from rdkit import Chem
    
    augmented_data = {}
    for smiles in smiles_list:
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                augmented_data[smiles] = [smiles]  # Keep original if parsing fails
                continue
                
            augmented = [smiles]
            for _ in range(num_augmentations):
                # Generate random SMILES for the same molecule
                rand_smiles = Chem.MolToSmiles(mol, doRandom=True, isomericSmiles=True)
                if rand_smiles != smiles and rand_smiles not in augmented:
                    augmented.append(rand_smiles)
            
            augmented_data[smiles] = augmented
        except Exception as e:
            print(f"Error augmenting SMILES {smiles}: {str(e)}")
            augmented_data[smiles] = [smiles]  # Fallback to original
            
    return augmented_data


class TPADataset(Dataset):
    def __init__(self, data_list, is_train=True, scaler=None, use_augmentation=False):
        """
        Enhanced TPADataset with SMILES augmentation option
        
        Args:
            data_list (list): List of data dictionaries
            is_train (bool): Whether dataset is for training
            scaler (StandardScaler): Optional pre-fitted scaler
            use_augmentation (bool): Whether to use SMILES augmentation
        """
        # Validate dataset first
        valid_data, report = validate_dataset(data_list)
        
        if len(valid_data) < len(data_list):
            print(f"Filtered {report['filtered']} invalid entries from dataset. Issues: {report['issues']}")
            
        if not valid_data:
            raise ValueError("No valid data entries after validation!")
        
        self.tokenizer = AutoTokenizer.from_pretrained(MOLFORMER_PATH, padding=True, trust_remote_code=True)
        self.original_data = valid_data
        self.is_train = is_train
        self.use_augmentation = use_augmentation and is_train  # Only use augmentation for training
        
        # Apply SMILES augmentation if enabled
        if self.use_augmentation:
            self.smiles_list = []
            self.augmented_indices = []  # Maps augmented index to original data index
            
            original_smiles = [item['smiles'] for item in valid_data]
            augmented_smiles_dict = smiles_augmentation(original_smiles)
            
            for idx, item in enumerate(valid_data):
                orig_smiles = item['smiles']
                for aug_smiles in augmented_smiles_dict.get(orig_smiles, [orig_smiles]):
                    self.smiles_list.append(aug_smiles)
                    self.augmented_indices.append(idx)
            
            print(f"Augmented dataset from {len(valid_data)} to {len(self.smiles_list)} samples")
        else:
            self.smiles_list = [item['smiles'] for item in valid_data]
            self.augmented_indices = list(range(len(valid_data)))

        # Create condition data using all numerical parameters
        self.condition_data = np.array([
            [
                valid_data[self.augmented_indices[i]]['wavelength'],
                valid_data[self.augmented_indices[i]]['ET(30)'],
                valid_data[self.augmented_indices[i]]['dielectic constant'],
                valid_data[self.augmented_indices[i]]['dipole moment']
            ]
            for i in range(len(self.smiles_list))
        ], dtype=np.float32)

        # Extract target values
        self.tpa_values = np.array([
            valid_data[self.augmented_indices[i]]['TPACS_log'] 
            for i in range(len(self.smiles_list))
        ], dtype=np.float32)
        
        # Handle scaling as before
        if is_train:
            if scaler is None:
                from sklearn.preprocessing import StandardScaler
                self.scaler = StandardScaler().fit(self.condition_data)
            else:
                self.scaler = scaler
            self.scaled_conditions = self.scaler.transform(self.condition_data)
        else:
            if scaler is not None:
                self.scaler = scaler
                self.scaled_conditions = self.scaler.transform(self.condition_data)
            else:
                self.scaled_conditions = self.condition_data
        
        # Pre-tokenize all SMILES
        self.tokenized_smiles = self.tokenizer(
            self.smiles_list,
            padding='max_length',
            truncation=True,
            max_length=512,
            return_tensors='pt'
        )

    def set_scaler(self, scaler):
        """
        Apply an external fitted scaler to the dataset's condition data.

        Args:
            scaler (StandardScaler): A fitted scaler to apply.
        """
        self.scaler = scaler
        self.scaled_conditions = self.scaler.transform(self.condition_data)

    def __len__(self):
        return len(self.smiles_list)

    def __getitem__(self, idx):
        """
        Returns a dictionary with model inputs and targets.
        """
        # Get tokenized SMILES
        input_ids = self.tokenized_smiles['input_ids'][idx]
        attention_mask = self.tokenized_smiles['attention_mask'][idx]

        # Get condition vector
        condition = torch.tensor(self.scaled_conditions[idx], dtype=torch.float32)

        # Get target value
        tpa = torch.tensor(self.tpa_values[idx], dtype=torch.float32).unsqueeze(0)

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'condition': condition,
            'labels': tpa
        }