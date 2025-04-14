import torch.nn as nn

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
    