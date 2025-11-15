import torch
import torch.nn as nn

from config import FEATURE_DIM, HIDDEN1, HIDDEN2, HIDDEN3, HIDDEN4, HIDDEN5, DROPOUT_RATE


class ResidualBlock(nn.Module):
    """
    Residual block with LayerNorm and Dropout.
    Architecture: Linear -> ReLU -> Dropout -> Linear + skip -> LayerNorm
    """
    def __init__(self, dim, dropout_rate=0.05):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, x):
        residual = x
        out = torch.relu(self.fc1(x))
        out = self.dropout(out)
        out = self.fc2(out)
        out = out + residual  # Skip connection
        out = self.norm(out)
        return out


class SimpleNNUE(nn.Module):
    """
    SimpleNNUE: Deep residual architecture for chess position evaluation.
    
    Architecture: 795 -> 2048 -> 2048 -> 1024 -> 512 -> 256 -> 1
    
    Features (795 total):
    - 768: Piece-square occupancies (64 squares × 12 piece types, perspective-flipped)
    - 1: Side-to-move (always 1.0 after perspective flip)
    - 4: Castling rights (our K, our Q, enemy K, enemy Q)
    - 8: En-passant file (one-hot encoding)
    - 6: Material balance per piece type
    - 6: Our piece counts
    - 1: Game phase (0=opening, 1=endgame)
    
    Each stage:
    - Linear + ReLU + LayerNorm + Dropout(0.05)
    - 2 Residual blocks
    
    Output: Scalar evaluation (no final activation)
    """

    def __init__(self):
        super().__init__()
        
        # Stage 1: 795 -> 2048
        self.fc1 = nn.Linear(FEATURE_DIM, HIDDEN1)
        self.norm1 = nn.LayerNorm(HIDDEN1)
        self.dropout1 = nn.Dropout(DROPOUT_RATE)
        self.res1_1 = ResidualBlock(HIDDEN1, DROPOUT_RATE)
        self.res1_2 = ResidualBlock(HIDDEN1, DROPOUT_RATE)
        
        # Stage 2: 2048 -> 2048
        self.fc2 = nn.Linear(HIDDEN1, HIDDEN2)
        self.norm2 = nn.LayerNorm(HIDDEN2)
        self.dropout2 = nn.Dropout(DROPOUT_RATE)
        self.res2_1 = ResidualBlock(HIDDEN2, DROPOUT_RATE)
        self.res2_2 = ResidualBlock(HIDDEN2, DROPOUT_RATE)
        
        # Stage 3: 2048 -> 1024
        self.fc3 = nn.Linear(HIDDEN2, HIDDEN3)
        self.norm3 = nn.LayerNorm(HIDDEN3)
        self.dropout3 = nn.Dropout(DROPOUT_RATE)
        self.res3_1 = ResidualBlock(HIDDEN3, DROPOUT_RATE)
        self.res3_2 = ResidualBlock(HIDDEN3, DROPOUT_RATE)
        
        # Stage 4: 1024 -> 512
        self.fc4 = nn.Linear(HIDDEN3, HIDDEN4)
        self.norm4 = nn.LayerNorm(HIDDEN4)
        self.dropout4 = nn.Dropout(DROPOUT_RATE)
        self.res4_1 = ResidualBlock(HIDDEN4, DROPOUT_RATE)
        self.res4_2 = ResidualBlock(HIDDEN4, DROPOUT_RATE)
        
        # Stage 5: 512 -> 256
        self.fc5 = nn.Linear(HIDDEN4, HIDDEN5)
        self.norm5 = nn.LayerNorm(HIDDEN5)
        self.dropout5 = nn.Dropout(DROPOUT_RATE)
        self.res5_1 = ResidualBlock(HIDDEN5, DROPOUT_RATE)
        self.res5_2 = ResidualBlock(HIDDEN5, DROPOUT_RATE)
        
        # Output head: 256 -> 1 (no activation)
        self.fc_out = nn.Linear(HIDDEN5, 1)

    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: (batch, 795) tensor of features
            
        Returns:
            (batch, 1) tensor of evaluations
        """
        # Stage 1
        x = torch.relu(self.fc1(x))
        x = self.norm1(x)
        x = self.dropout1(x)
        x = self.res1_1(x)
        x = self.res1_2(x)
        
        # Stage 2
        x = torch.relu(self.fc2(x))
        x = self.norm2(x)
        x = self.dropout2(x)
        x = self.res2_1(x)
        x = self.res2_2(x)
        
        # Stage 3
        x = torch.relu(self.fc3(x))
        x = self.norm3(x)
        x = self.dropout3(x)
        x = self.res3_1(x)
        x = self.res3_2(x)
        
        # Stage 4
        x = torch.relu(self.fc4(x))
        x = self.norm4(x)
        x = self.dropout4(x)
        x = self.res4_1(x)
        x = self.res4_2(x)
        
        # Stage 5
        x = torch.relu(self.fc5(x))
        x = self.norm5(x)
        x = self.dropout5(x)
        x = self.res5_1(x)
        x = self.res5_2(x)
        
        # Output (no activation)
        x = self.fc_out(x)
        return x

