from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer
from constants import MOLFORMER_PATH


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

class TPADataset(Dataset):
    def __init__(self, data_list, is_train=True, scaler=None):
        """
        Dataset for two-photon absorption prediction without using a tokenizer.
        All numerical parameters (e.g. wavelength and solvent properties) are combined
        into a single condition vector.

        Args:
            data_list (list): List of dictionaries. Each dictionary should contain:
                              - "smiles": SMILES string (retained for reference)
                              - "wavelength": Excitation wavelength (numerical)
                              - "ET(30)": A solvent property (numerical)
                              - "dielectic constant": A solvent property (numerical)
                              - "dipole moment": A solvent property (numerical)
                              - "TPACS_log" or "TPACS": The target TPA value 
            is_train (bool): Whether the dataset is for training (scaling is fitted) or not.
            scaler (StandardScaler or None): Optionally provide a pre-fitted scaler.
        """
        # Validate dataset first
        valid_data, report = validate_dataset(data_list)
       
        if len(valid_data) < len(data_list):
            print(f"Filtered {report['filtered']} invalid entries from dataset. Issues: {report['issues']}")
           
        if not valid_data:
            raise ValueError("No valid data entries after validation!")
       
        self.tokenizer = AutoTokenizer.from_pretrained(MOLFORMER_PATH, padding=True, trust_remote_code=True)
        self.data = valid_data
        self.is_train = is_train

        self.smiles_list = [item['smiles'] for item in valid_data]

        # Tokenize SMILES strings
        self.smiles_tokens = self.tokenizer(
            self.smiles_list,
            padding=True,
            return_tensors='pt'
        )

        # Create condition data using all numerical parameters.
        self.condition_data = np.array([
            [
                item['wavelength'],
                item['ET(30)'],
                item['dielectic constant'],
                item['dipole moment']
            ]
            for item in data_list
        ], dtype=np.float32)

        # Extract target values. Use 'TPACS_log'
        self.tpa_values = np.array([item['TPACS_log'] for item in data_list], dtype=np.float32)

        # Setup the scaler for the condition data
        if is_train:
            # If no scaler is provided, fit a new one on the condition data.
            if scaler is None:
                from sklearn.preprocessing import StandardScaler
                self.scaler = StandardScaler().fit(self.condition_data)
            else:
                self.scaler = scaler
            self.scaled_conditions = self.scaler.transform(self.condition_data)
        else:
            # For validation or test, either a scaler must be provided or it must be set later.
            if scaler is not None:
                self.scaler = scaler
                self.scaled_conditions = self.scaler.transform(self.condition_data)
            else:
                # Use raw condition values until an external scaler is applied.
                self.scaled_conditions = self.condition_data

    def set_scaler(self, scaler):
        """
        Apply an external fitted scaler to the dataset's condition data.

        Args:
            scaler (StandardScaler): A fitted scaler to apply.
        """
        self.scaler = scaler
        self.scaled_conditions = self.scaler.transform(self.condition_data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Returns a dictionary with:
            - 'input_ids': tokenized SMILES string
            - 'attention_mask': Attention mask for the tokenized SMILES
            - 'condition': Scaled condition tensor including all numerical parameters
            - 'tpa': Target TPA value as a tensor
        """
        # Convert the condition vector to torch tensor
        condition = torch.tensor(self.scaled_conditions[idx], dtype=torch.float32)

        # Get target value
        tpa = torch.tensor(self.tpa_values[idx], dtype=torch.float32)

        
        smiles_token = self.smiles_tokens[idx]
        input_ids = smiles_token.ids
        attention_mask = smiles_token.attention_mask

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'condition': condition,
            'labels': tpa
        }
