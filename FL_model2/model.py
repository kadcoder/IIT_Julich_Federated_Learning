import torch
import torch.nn as nn
import config

class AgePredictor(torch.nn.Module):
    def __init__(self, input_size, dropout_rate=config.DROPOUT_RATE):
        super().__init__()
        
        # First Layer
        self.fc1 = nn.Linear(input_size, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.act1 = nn.LeakyReLU(0.02)
        self.drop1 = nn.Dropout(dropout_rate)
        
        # Second Layer
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.act2 = nn.LeakyReLU(0.02)
        self.drop2 = nn.Dropout(dropout_rate)
        
        # Skip Connection Layer
        #self.skip = nn.Linear(input_size, 256) if input_size != 256 else nn.Identity()

        # Output Layer
        self.fc3 = nn.Linear(256, 1)
    
    def initialize_weights(self, model):
        for m in model.modules():
            if isinstance(m, nn.Linear):
                # Kaiming He initialization for LeakyReLU activation
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu', a=0.01)
                nn.init.constant_(m.bias, 0.01)  # Small positive bias to prevent dead neurons
                
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)  # Scale factor
                nn.init.constant_(m.bias, 0)    # Shift factor
    
    def forward(self, x):
        x = self.drop1(self.act1(self.bn1(self.fc1(x))))
        x = self.drop2(self.act2(self.bn2(self.fc2(x))))
        x = self.fc3(x)
        return x