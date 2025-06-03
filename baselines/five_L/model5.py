import torch
import torch.nn as nn

class AgePredictor(nn.Module):
    """
    A fully connected neural network for age prediction from tabular feature data with four hidden layers.
    
    Architecture Details:
    - Input layer
    - Four hidden layers with BatchNorm, LeakyReLU, and Dropout
    - Output regression layer
    
    Args:
        input_size (int): Number of input features.
        dropout_rate (float): Dropout probability applied after each hidden layer. Default: 0.2.
        
    Shape:
        - Input: (batch_size, input_size)
        - Output: (batch_size, 1)
        
    Example:
        >>> model = AgePredictor(input_size=256)
        >>> x = torch.randn(32, 256)
        >>> output = model(x)
    """
    
    def __init__(self, input_size: int, dropout_rate: float = 0.2):
        super().__init__()
        
        # First hidden layer
        self.fc1 = nn.Linear(input_size, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.act1 = nn.LeakyReLU(0.01)
        self.drop1 = nn.Dropout(dropout_rate)
        
        # Second hidden layer
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.act2 = nn.LeakyReLU(0.01)
        self.drop2 = nn.Dropout(dropout_rate)
        
        # Third hidden layer
        self.fc3 = nn.Linear(256, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.act3 = nn.LeakyReLU(0.01)
        self.drop3 = nn.Dropout(dropout_rate)
        
        # Fourth hidden layer (newly added)
        self.fc4 = nn.Linear(128, 64)
        self.bn4 = nn.BatchNorm1d(64)
        self.act4 = nn.LeakyReLU(0.01)
        self.drop4 = nn.Dropout(dropout_rate)
        
        # Output layer
        self.fc5 = nn.Linear(64, 1)

    def initialize_weights(self, m: nn.Module) -> None:
        """
        Initializes weights using Kaiming Normal initialization for linear layers
        and constant initialization for biases. For BatchNorm layers, initializes
        weights to 1 and biases to 0.
        
        Args:
            m (nn.Module): Layer module to initialize
        """
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu', a=0.01)
            nn.init.constant_(m.bias, 0.01)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass of the network.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_size)
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, 1) containing predicted age values
        """
        x = self.drop1(self.act1(self.bn1(self.fc1(x))))
        x = self.drop2(self.act2(self.bn2(self.fc2(x))))
        x = self.drop3(self.act3(self.bn3(self.fc3(x))))
        x = self.drop4(self.act4(self.bn4(self.fc4(x))))  # New fourth hidden layer
        x = self.fc5(x)
        return x