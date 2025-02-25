import pandas as pd
import numpy as np
from copy import deepcopy
import pickle
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import NAdam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class AgePredictor(nn.Module):
    def __init__(self, input_size, dropout_rate=0.3, LeakyReLu_slope=0.01):
        super(AgePredictor, self).__init__()
        self.fc1 = nn.Linear(input_size, 1024)
        self.bn1 = nn.BatchNorm1d(1024)
        self.fc2 = nn.Linear(1024, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.fc3 = nn.Linear(512, 1)
        self.leakyrelu = nn.LeakyReLU(negative_slope=LeakyReLu_slope)
        self.dropout = nn.Dropout(p=dropout_rate)

        # Initialize weights
        self._initialize_weights()

    def forward(self, x):
        x = self.leakyrelu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = self.leakyrelu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if m is self.fc3:
                    nn.init.xavier_uniform_(m.weight, gain=1.0)
                    nn.init.zeros_(m.bias)
                else:
                    nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu', a=0.02)
                    nn.init.normal_(m.bias, mean=0.0, std=0.01)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                
# Custom Dataset for PyTorch
class AgePredictionDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

    
class Train_Model():
    
    def preprocess_data(df,target_column):
        '''
        df: Input dataframe 
        target_column: Column containing target variables
        '''
        # Separate features and labels
        X = df.drop(columns=[target_column])  # Assuming 'age' is the target column
        y = df[target_column]

        # Split into training, validation, and test sets
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42) #original test_size=0.2
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.4, random_state=42) #original test_size=0.4

        # Normalize the features (scaling them to have 0 mean and 1 standard deviation)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)

        # Convert to PyTorch tensors
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
        
        y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
        y_val_tensor = torch.tensor(y_val.values, dtype=torch.float32).view(-1, 1)
        y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

        return X_train_tensor, X_val_tensor, X_test_tensor, y_train_tensor, y_val_tensor, y_test_tensor
    
    def train_model(model, X_train, X_val, X_test, y_train, y_val, y_test, epochs=500, learning_rate=0.00005, device=None, weights_path='best_model_weights.pth'):
        model = model.to(device)
        
        train_dataset = AgePredictionDataset(X_train, y_train)
        val_dataset = AgePredictionDataset(X_val, y_val)
        test_dataset = AgePredictionDataset(X_test, y_test)

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        
        criterion = nn.L1Loss()
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=1e-3)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=20, verbose=True)

        train_losses, val_losses = [], []
        patience_model_reload = 25
        best_val_loss = float('inf')
        epochs_without_improvement_model_reload = 0
        best_model_weights = None


        for epoch in range(epochs):
            model.train()
            total_train_loss = 0
            batch_gradients = {}

            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                predictions = model(inputs)
                loss = criterion(predictions, targets)
                total_train_loss += loss.item()

                optimizer.zero_grad()
                loss.backward()

                for name, param in model.named_parameters():
                    if param.grad is not None:
                        if name not in batch_gradients:
                            batch_gradients[name] = []
                        batch_gradients[name].append(param.grad.clone().detach()) #append layer-wise gradients to this dictionary for each epoch

                optimizer.step()

            avg_train_loss = total_train_loss / len(train_loader)
            train_losses.append(avg_train_loss)

            model.eval()
            total_val_loss = 0
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    predictions = model(inputs)
                    loss = criterion(predictions, targets)
                    total_val_loss += loss.item()

            avg_val_loss = total_val_loss / len(val_loader)
            val_losses.append(avg_val_loss)

            print(f"Epoch {epoch+1}/{epochs}, Train MAE Loss: {avg_train_loss:.4f}, Val MAE Loss: {avg_val_loss:.4f}")

            scheduler.step(avg_val_loss)

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_model_weights = model.state_dict()
                torch.save(best_model_weights, weights_path)
                epochs_without_improvement_model_reload = 0
            
            else:
                epochs_without_improvement_model_reload += 1

            if epochs_without_improvement_model_reload >= patience_model_reload and best_model_weights is not None:
                print("Restoring the best model weights due to early stopping.")
                model.load_state_dict(torch.load(weights_path))
                epochs_without_improvement_model_reload = 0

        model.eval()
        total_test_loss = 0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                predictions = model(inputs)
                loss = criterion(predictions, targets)
                total_test_loss += loss.item()

        avg_test_loss = total_test_loss / len(test_loader)
        print(f"Test MAE Loss: {avg_test_loss:.4f}")

        return train_losses, val_losses, avg_test_loss


    
    