import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
from torch.amp import autocast
import os
from encoder import ConditionEncoder, AttentionFusion, TPAPredictionHead
from constants import MOLFORMER_PATH, DEVICE

# Set environment variable to avoid tokenizers warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class MolFormerTPA(nn.Module):
    def __init__(self, condition_dim=4, use_mixed_precision=False):
        """
        Improved model for predicting two-photon absorption
        
        Args:
            condition_dim (int): Dimension of conditions
            use_mixed_precision (bool): Whether to use mixed precision
        """
        super().__init__()

        # SMILES Encoding with MolFormer
        self.molformer = AutoModel.from_pretrained(MOLFORMER_PATH, trust_remote_code=True)
        molformer_dim = self.molformer.config.hidden_size
        self.use_mixed_precision = use_mixed_precision
        
        # Pooling for MolFormer outputs (mean pooling instead of just CLS token)
        self.pool_type = "mean"  # Options: "cls", "mean", "max", "attention"
        
        # If using attention pooling, we need an attention layer
        if self.pool_type == "attention":
            self.attention_pooling = nn.Sequential(
                nn.Linear(molformer_dim, 512),
                nn.Tanh(),
                nn.Linear(512, 1)
            )

        # Conditions Encoding with improved architecture
        self.condition_encoder = ConditionEncoder(
            condition_dim=condition_dim,
            hidden_dim=128,
            output_dim=256
        )
        
        # Enhanced fusion with self-attention
        self.fusion = AttentionFusion(
            mol_dim=molformer_dim,
            cond_dim=256,
            output_dim=512
        )
        
        # Improved prediction head
        self.prediction_head = TPAPredictionHead(input_dim=512)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights with improved method"""
        for module in [self.condition_encoder, self.fusion, self.prediction_head]:
            for name, param in module.named_parameters():
                if 'weight' in name and param.dim() > 1:
                    # Kaiming He initialization for weights
                    nn.init.kaiming_normal_(param, mode='fan_out', nonlinearity='relu')
                elif 'bias' in name:
                    # Zero initialization for biases
                    nn.init.zeros_(param)
                    
    def _pool_molformer_output(self, last_hidden_state, attention_mask):
        """
        Pool MolFormer outputs with various strategies
        
        Args:
            last_hidden_state: MolFormer hidden states [batch, seq_len, hidden_dim]
            attention_mask: Attention mask [batch, seq_len]
            
        Returns:
            torch.Tensor: Pooled representation [batch, hidden_dim]
        """
        if self.pool_type == "cls":
            # Use [CLS] token
            return last_hidden_state[:, 0]
        
        elif self.pool_type == "mean":
            # Mean pooling
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(
                last_hidden_state.size()
            ).float()
            sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
            sum_mask = input_mask_expanded.sum(1)
            sum_mask = torch.clamp(sum_mask, min=1e-9)
            return sum_embeddings / sum_mask
        
        elif self.pool_type == "max":
            # Max pooling
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(
                last_hidden_state.size()
            ).float()
            # Set padding tokens to large negative value
            embeddings = last_hidden_state.clone()
            embeddings[input_mask_expanded == 0] = -1e9
            return torch.max(embeddings, dim=1)[0]
        
        elif self.pool_type == "attention":
            # Attention pooling
            attention_scores = self.attention_pooling(last_hidden_state).squeeze(-1)
            attention_mask = attention_mask.float()
            # Mask out padding tokens
            attention_scores = attention_scores.masked_fill(attention_mask == 0, -1e9)
            attention_weights = F.softmax(attention_scores, dim=1)
            # Apply attention weights
            return torch.sum(last_hidden_state * attention_weights.unsqueeze(-1), dim=1)
            
    def change_molformer(self, train_molformer=True):
        """Control whether to train the MolFormer backbone"""
        for param in self.molformer.parameters():
            param.requires_grad = train_molformer

    def forward(self, input_ids, attention_mask, condition, labels=None):
        """Forward pass with improved architecture"""
        # Use mixed precision for MolFormer if configured
        with autocast(device_type=DEVICE.type, enabled=self.use_mixed_precision):
            molformer_output = self.molformer(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True
            )
        
        # Pool MolFormer outputs using selected strategy
        mol_repr = self._pool_molformer_output(
            molformer_output.last_hidden_state, 
            attention_mask
        )
        
        # Encode conditions
        cond_repr = self.condition_encoder(condition)
        
        # Fuse representations using attention mechanism
        fused = self.fusion(mol_repr, cond_repr)
        
        # Make prediction
        tpa_prediction = self.prediction_head(fused)
        
        # Calculate loss if labels provided
        if labels is not None:
            # MSE loss with additional L1 regularization
            mse_loss = F.mse_loss(tpa_prediction, labels)
            l1_loss = F.l1_loss(tpa_prediction, labels)
            loss = mse_loss + 0.1 * l1_loss
            return {"loss": loss, "predictions": tpa_prediction}
            
        return {"predictions": tpa_prediction}


class ModelEnsemble:
    def __init__(self, model_class, model_paths, device=DEVICE):
        """
        Ensemble of MolFormerTPA models
        
        Args:
            model_class: Model class (MolFormerTPA or ImprovedMolFormerTPA)
            model_paths: List of paths to trained model weights
            device: Device to run inference on
        """
        self.models = []
        self.device = device
        
        # Load all models
        for path in model_paths:
            model = model_class(use_mixed_precision=True).to(device)
            model.load_state_dict(torch.load(path, map_location=device))
            model.eval()
            self.models.append(model)
            
        print(f"Loaded ensemble of {len(self.models)} models")
    
    def predict(self, input_ids, attention_mask, condition):
        """
        Make ensemble prediction
        
        Args:
            input_ids: Tokenized SMILES input
            attention_mask: Attention mask
            condition: Condition tensor
            
        Returns:
            torch.Tensor: Ensemble prediction
        """
        predictions = []
        
        with torch.no_grad():
            for model in self.models:
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    condition=condition
                )
                predictions.append(outputs["predictions"])
        
        # Calculate mean prediction and uncertainty
        ensemble_pred = torch.stack(predictions).mean(dim=0)
        uncertainty = torch.stack(predictions).std(dim=0)
        
        return {
            "predictions": ensemble_pred,
            "uncertainty": uncertainty
        }


def predict_tpacs(model, smiles_list, wavelength, et30, dielectric, dipole, 
                 batch_size=32, scaler=None):
    """
    Batch prediction function for TPA cross-sections
    
    Args:
        model: Trained MolFormerTPA model
        smiles_list: List of SMILES strings
        wavelength: Excitation wavelength
        et30: ET(30) solvent parameter
        dielectric: Dielectric constant
        dipole: Dipole moment
        batch_size: Batch size for prediction
        scaler: Scaler for condition normalization
        
    Returns:
        numpy.ndarray: Predicted TPA cross-sections (log scale)
    """
    from transformers import AutoTokenizer
    from constants import MOLFORMER_PATH
    import numpy as np
    
    # Set model to evaluation mode
    model.eval()
    
    # Create tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MOLFORMER_PATH, trust_remote_code=True)
    
    # Prepare condition data
    conditions = np.array([[wavelength, et30, dielectric, dipole]] * len(smiles_list))
    
    # Scale conditions if scaler provided
    if scaler is not None:
        conditions = scaler.transform(conditions)
    
    # Convert conditions to tensor
    condition_tensor = torch.tensor(conditions, dtype=torch.float32).to(DEVICE)
    
    # Tokenize all SMILES at once with padding
    inputs = tokenizer(
        smiles_list,
        padding='max_length',
        truncation=True,
        max_length=512,
        return_tensors='pt'
    )
    
    # Move input tensors to device
    input_ids = inputs['input_ids'].to(DEVICE)
    attention_mask = inputs['attention_mask'].to(DEVICE)
    
    # Predict in batches
    predictions = []
    with torch.no_grad():
        for i in range(0, len(smiles_list), batch_size):
            # Get batch
            batch_input_ids = input_ids[i:i+batch_size]
            batch_attention_mask = attention_mask[i:i+batch_size]
            batch_condition = condition_tensor[i:i+batch_size]
            
            # Forward pass
            outputs = model(
                input_ids=batch_input_ids,
                attention_mask=batch_attention_mask,
                condition=batch_condition
            )
            
            # Collect predictions
            batch_preds = outputs["predictions"].cpu().numpy()
            predictions.extend(batch_preds)
    
    return np.array(predictions)