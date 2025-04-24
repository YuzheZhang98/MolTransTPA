import torch
import torch.nn as nn
import torch.nn.functional as F


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
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, output_dim)
        )

    def forward(self, condition):
        return self.mlp(condition)


class AttentionFusion(nn.Module):
    def __init__(self, mol_dim, cond_dim, output_dim):
        """
        Self-attention fusion layer to combine molecule and condition features
        
        Args:
            mol_dim (int): Dimension of molecule representation
            cond_dim (int): Dimension of condition representation
            output_dim (int): Output dimension
        """
        super().__init__()
        
        # Project to same dimension for attention
        self.mol_proj = nn.Linear(mol_dim, output_dim)
        self.cond_proj = nn.Linear(cond_dim, output_dim)
        
        # Attention mechanism
        self.query = nn.Linear(output_dim, output_dim)
        self.key = nn.Linear(output_dim, output_dim)
        self.value = nn.Linear(output_dim, output_dim)
        
        # Output projection
        self.output_proj = nn.Linear(output_dim, output_dim)
        self.layer_norm = nn.LayerNorm(output_dim)
        
    def forward(self, mol_repr, cond_repr):
        # Project features to common space
        mol_feat = self.mol_proj(mol_repr)
        cond_feat = self.cond_proj(cond_repr)
        
        # Concatenate features for attention
        combined = torch.stack([mol_feat, cond_feat], dim=1)  # [batch, 2, dim]
        
        # Self-attention
        q = self.query(combined)
        k = self.key(combined)
        v = self.value(combined)
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / (combined.size(-1) ** 0.5)
        attn_weights = F.softmax(scores, dim=-1)
        
        # Apply attention
        attn_output = torch.matmul(attn_weights, v)
        
        # Sum attention outputs and apply final projection
        fused = attn_output.sum(dim=1)
        output = self.output_proj(fused)
        
        # Residual connection and layer norm
        output = self.layer_norm(output + mol_feat)
        
        return output


class TPAPredictionHead(nn.Module):
    def __init__(self, input_dim, hidden_dim=256):
        """
        Improved prediction head for TPA cross-section

        Args:
            input_dim (int): Input dimension
            hidden_dim (int): Hidden dimension
        """
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.LayerNorm(hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1)
        )

    def forward(self, x):
        return self.mlp(x)