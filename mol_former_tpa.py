import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
from torch.amp import autocast
from encoder import ConditionEncoder, TPAPredictionHead
from constants import MOLFORMER_PATH, DEVICE

class MolFormerTPA(nn.Module):
    def __init__(self, condition_dim=4, use_mixed_precision=False):
        """
        Model for predicting two-photon absorption using MolFormer
        
        Args:
            condition_dim (int): Dimension of conditions (wavelength + solvent properties)
            use_mixed_precision (bool): Whether to use mixed precision for inference
        """
        super().__init__()

        # 1. SMILES Encoding Branch - MolFormer
        self.molformer = AutoModel.from_pretrained(MOLFORMER_PATH, trust_remote_code=True)
        molformer_dim = self.molformer.config.hidden_size
        self.use_mixed_precision = use_mixed_precision

        # 2. Conditions Encoding Branch
        self.condition_encoder = ConditionEncoder(
            condition_dim=condition_dim,
            output_dim=128
        )

        # 3. Fusion Layer
        self.fusion = nn.Linear(molformer_dim + 128, 256)

        # 4. Prediction Head
        self.prediction_head = TPAPredictionHead(input_dim=256)

    def change_molformer(self, train_molformer=True):
        """Control whether to train the MolFormer backbone or keep it frozen"""
        for param in self.molformer.parameters():
            param.requires_grad = train_molformer

    def forward(self, input_ids, attention_mask, condition, labels=None):
        """Forward pass of the model
        
        Args:
            input_ids: Tokenized SMILES inputs
            attention_mask: Attention mask for the tokenized inputs
            condition: Condition tensor (wavelength + solvent properties)
            labels: Optional ground truth labels for loss calculation
            
        Returns:
            dict: Dictionary containing predictions and loss (if labels provided)
        """
        # 1. SMILES Encoding with optional mixed precision
        # Use mixed precision based on model configuration
        with autocast(device_type=DEVICE.type, enabled=self.use_mixed_precision):
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

        # 5. Loss Calculation (if labels provided)
        if labels is not None:
            loss = F.mse_loss(tpa_prediction, labels)
            return {"loss": loss, "predictions": tpa_prediction}
        return {"predictions": tpa_prediction}