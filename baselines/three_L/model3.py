import torch
import torch.nn as nn
from typing import Dict

class AgePredictor(nn.Module):
    
    #Add a detailed docstring here.
    #Example:
    #A fully connected neural network for age prediction from tabular feature data.
    
    #Args:
    #    input_size (int): Number of input features.
    #    dropout_rate (float): Dropout probability applied after each hidden layer.
    
    
    def __init__(self, input_size: int, dropout_rate: float = 0.2):  #Add more detailed type hints if needed
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
        
        # Output layer
        self.fc3 = nn.Linear(256, 1)

    def initialize_weights(self, m: nn.Module) -> None:  #Add return type and parameter docstring
        
        #Add a description of what this function does.
        #Initializes weights using Kaiming Normal and constants for biases.
        
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu', a=0.01)
                nn.init.constant_(m.bias, 0.01)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # Describe input/output shapes
        
        #Add forward pass documentation.
        
        x = self.drop1(self.act1(self.bn1(self.fc1(x))))
        x = self.drop2(self.act2(self.bn2(self.fc2(x))))
        x = self.fc3(x)
        return x
