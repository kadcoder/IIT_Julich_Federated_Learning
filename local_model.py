import torch
import os,sys
import argparse
import math
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from copy import deepcopy
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
import harmonize as hrm


# Configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEED = 42
torch.manual_seed(SEED)
#np.random.seed(SEED)

# Global flag to track initialization
WEIGHTS_INITIALIZED = False
DROPOUT_RATE = 0.2

# Hyperparameters
BATCH_SIZE = 32
INIT_LR = 1e-3
L1_LAMBDA = 0.001
WEIGHT_DECAY = 1e-5
PATIENCE = 20
MAX_EPOCHS = 5
NUM_EPOCHS = 1
train_ratio = 0.7

class AgePredictor(nn.Module):
    def __init__(self, input_size, dropout_rate=DROPOUT_RATE):
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
        
        # Output Layer
        #self.bn3 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, 1)
        
        global WEIGHTS_INITIALIZED
        if not WEIGHTS_INITIALIZED:  
            self._initialize_weights()  
            WEIGHTS_INITIALIZED = True  # Mark as initialized globally

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if m is self.fc3:
                    # Use Xavier Normal with a slightly higher gain to increase variance
                    nn.init.xavier_normal_(m.weight, gain=torch.sqrt(torch.tensor(2.0)))
                    nn.init.normal_(m.bias, mean=0.0, std=0.1)  # Ensure non-zero bias
                    #nn.init.zeros_(m.bias)
                else:
                    nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='leaky_relu', a=0.02)
                    nn.init.normal_(m.bias, mean=0.0, std=0.1)
                #nn.init.zeros_(m.bias)

            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.drop1(self.act1(self.bn1(self.fc1(x))))
        x = self.drop2(self.act2(self.bn2(self.fc2(x))))
        x = self.fc3(x)
        return x

def preprocess(data_file_path):

    silo_name = os.path.basename(os.path.dirname(data_file_path))
    data = pd.read_csv(data_file_path)

    # Get feature columns (assuming columns are named '1', '2', ..., '1073')
    feature_columns = [str(i) for i in range(1, 1074)]  # Generates ['1', '2', ..., '1073']

    # Convert to numpy arrays
    X = data[feature_columns].to_numpy().astype(np.float32)  #Features matrix
    y = data['age'].to_numpy().astype(np.float32)            #Target vector

    print(f"Silo name : {silo_name}")

    # Verify shapes
    print(f"Feature matrix shape: {X.shape}")  # Should be (n_samples, 1073)
    print(f"Target vector shape: {y.shape}")   # Should be (n_samples,)

    return X,y,silo_name

def extractGradients(model):

    gradients = {}
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            grad = param.grad.clone().detach()
            num_elements = grad.numel()

            # Compute gradient statistics
            mean = grad.mean().item()
            max_val = grad.max().item()
            min_val = grad.min().item()
            std = grad.std(unbiased=True).item() if num_elements > 1 else 0.0
            
            # Convert to NumPy for standardization
            grad_data = grad.cpu().numpy().copy()

            # Standardize: (data - mean) / std (avoid division by zero)
            if std > 0:
                grad_data = (grad_data - mean) / std + 1e-4
            else:
                grad_data = (grad_data - mean) + np.random.normal(0, 1e-4, grad_data.shape)  # If std=0, just subtract mean
                

            # Store results
            gradients[name] = {
                'mean': mean,
                'max': max_val,
                'min': min_val,
                'std': std,
                'data': grad_data
            }

    return gradients

def train_localmodel(train_path):

    X, y,silo_name = preprocess(train_path)
    # Compute split index
    train_size = int(len(X) * train_ratio)
    indices = np.random.permutation(len(X)) # Shuffle indices

    train_idx, val_idx = indices[:train_size], indices[train_size:]  # Split indices
    
    X_train, X_val = X[train_idx], X[val_idx]  # Assign train and validation sets
    y_train, y_val = y[train_idx], y[val_idx]

    scaler = StandardScaler()  # Normalization
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    print(f"train dataset shape: {X_train.shape}")  # Should be (n_samples, 1073)
    print(f"validation dataset shape: {X_val.shape}")   # Should be (n_samples, 1073)
    
    # Convert to tensors
    train_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train)),
        batch_size=BATCH_SIZE, shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val)),
        batch_size=BATCH_SIZE
    )

    model = AgePredictor(X_train.shape[1])
    model.to(DEVICE)

    criterion = nn.L1Loss()
    optimizer = optim.SGD(model.parameters(), lr = INIT_LR,weight_decay=WEIGHT_DECAY)


    train_losses, val_losses = [], []

    for epoch in range(MAX_EPOCHS):
        model.train()
        total_train_loss = 0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
        
            # Ensure target shape is (batch_size, 1)
            y_batch = y_batch.view(-1, 1)
            optimizer.zero_grad()
            predictions = model(X_batch)

            # Ensure predictions match target shape
            predictions = predictions.view(-1, 1)

            loss = criterion(predictions, y_batch)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        train_loss = total_train_loss / len(train_loader)
        train_losses.append(train_loss)

        # Validation
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for X_val, y_val in val_loader:
                X_val, y_val = X_val.to(DEVICE), y_val.to(DEVICE)
                # Ensure target shape
                y_val = y_val.view(-1, 1)

                val_predictions = model(X_val)
                val_predictions = val_predictions.view(-1, 1)

                loss = criterion(val_predictions, y_val)
                total_val_loss += loss.item()

        val_loss = total_val_loss / len(val_loader)
        val_losses.append(val_loss)

        #print(f"Epoch {epoch+1}/{self.max_epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    # Save the final model weights
    torch.save(model.state_dict(), f"/home/tanurima/germany/brain_age_parcels/m2/{silo_name}_{MAX_EPOCHS}.pth")
    print(f"Model weights saved as {silo_name}.pth")

    local_gradients = extractGradients(model)

    return local_gradients, scaler, optimizer

def train_globalmodel(train_path, scaler,optimizer,silo_name):

    print(f"fine-tuning model '{silo_name}' with global dataset...")

    try:
        X_train,y_train,_ = preprocess(train_path)

        # Define the model path
        model_path = f'/home/tanurima/germany/brain_age_parcels/m2/{silo_name}_{MAX_EPOCHS}.pth'

        # Load the model (assuming model_name corresponds to a model function or class)
        model = AgePredictor(X_train.shape[1])
        model.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=True))
        model.to(DEVICE)
        model.eval()
        
        X_train = scaler.transform(X_train)

        # Convert to tensors
        test_loader = DataLoader(
            TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train)),
            batch_size=BATCH_SIZE, shuffle=True
        )
        
        all_actuals = [] # Initialize lists to store actual vs predicted values for MAE calculation
        all_preds = []
        
        # Fine-tuning loop
        total_loss = 0.0
        num_batches = 0
        criterion = nn.L1Loss()
        #optimizer = optim.Adam(model.parameters(), lr=1e-3)
        
        # Iterate over global dataset
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            # Ensure target shape
            y_batch = y_batch.view(-1, 1)
            
            # Zero gradients and forward pass
            optimizer.zero_grad()
            predictions = model(X_batch).view(-1,1)
            loss = criterion(predictions, y_batch)
            
            # Backward pass with gradient clipping
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # Accumulate loss
            total_loss += loss.item()
            num_batches += 1
            
            # Collect actual and predicted values for MAE calculation
            all_actuals.extend(y_batch.cpu().numpy())
            all_preds.extend(predictions.cpu().detach().numpy())

        # Calculate average loss
        avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')
        #print(f'\nTest error of eNki :{avg_loss}\n')

        # Save the final model weights
        torch.save(model.state_dict(), f"/home/tanurima/germany/brain_age_parcels/m2/global_{MAX_EPOCHS}.pth")
        print(f"Model weights saved as {silo_name}.pth")

        global_gradients = extractGradients(model)
        
        if os.path.exists(model_path):
            os.remove(model_path)
            print(f"Deleted: {model_path}")

        return global_gradients
        
    except Exception as e:
        print(f"Error during fine-tuning: {e}")
        return float('inf')

def update_model_weights(model_id, harmonized_results, learning_rate=1e-3):

    model_state = model.state_dict()
    
    for layer_name, harmonized_grad in harmonized_results.items():
        if layer_name not in model_state:
            continue
        
        # Convert harmonized gradient to tensor
        grad_tensor = torch.from_numpy(harmonized_grad).float().to(DEVICE)
        
        # Update weights: w = w - lr * gradient
        model_state[layer_name] -= learning_rate * grad_tensor
    
    model.load_state_dict(model_state)
    return model


def train_silo(silo_name,global_name ='eNki'):

    # Load and preprocess data
    local_path = f'/home/tanurima/germany/brain_age_parcels/{silo_name}/{silo_name}_features_cleaned_train_data.csv'
    test_path = f'/home/tanurima/germany/brain_age_parcels/{global_name}/{global_name}_features_cleaned_train_data.csv'
    
    
    local_gradients,scaler,optimizer = train_localmodel(local_path)
    global_gradients = train_globalmodel(test_path, scaler,optimizer,silo_name)

    layers = list(local_gradients.keys())
    print(f'layers in the each silo {silo_name} :{layers}')
    for layer_name in local_gradients.keys():
        print(f"Layer: {layer_name}")
        print(f"Local Gradient  mean : {local_gradients[layer_name]['mean']} and std : {local_gradients[layer_name]['std']}")
        print(f"Global Gradient mean : {global_gradients[layer_name]['mean']} and std : {global_gradients[layer_name]['std']}\n")

    return hrm.harmonize_localsilo(layers, local_gradients, global_gradients)


# Example usage
if __name__ == '__main__':

    # Dummy data for demonstration
    """Main function to parse arguments and start federated training."""
    #parser = argparse.ArgumentParser(description="Federated Learning for Age Prediction")
    #parser.add_argument('--model-id', type=int, default=2, help="Model ID to use (2-7)")

    #args = parser.parse_args()
    #model_id = int(args.model_id)
    silos = {0: 'CamCAN',1: 'SALD'}


    for epoch in range(NUM_EPOCHS):  # Training loop

        print(f"\n=== Global Epoch {epoch+1}/{NUM_EPOCHS} ===")
        
        results = {}
        layers = []

        # 1. Train model and get gradients
        for i in list(silos.keys()):
            silo = silos[i]
            #print(f'Silo name :{silo}')
            results[silo] = train_silo(silo)

            for layer_name, grad in results[silo].items():
                print(f"Layer: {layer_name}, Gradient shape: {grad.shape}")
                #print(f"Gradient: {grad}")
                layers.append(layer_name)

        # 2. Harmonize gradients
        print(f'layers of the model:{layers}')
        harmonized_results = hrm.harmonize_global(layers,list(silos.values()), results)

        #for layer_name, grad in harmonized_results.items():
        #    print(f"Layer: {layer_name}, Gradient Mean: {grad.mean()}")

        # 3. Update model weights
        #model = update_model_weights(model, harmonized_results)
    


    """
    # Load data
    data_file_path = '/home/tanurima/germany/brain_age_parcels/CamCAN/CamCAN_features_cleaned.csv'
    X,y,silo_name = preprocess(data_file_path)

    local_gradients, global_gradients = train_model(X, y, silo_name)

    print('\n')

    layers = list(local_gradients.keys())
    print(f"layers of the model:{layers}")
    results = harmonize_all_gradients(layers,local_gradients,global_gradients,silo_name)
    # Update model weights
    updated_model = update_model_weights(model=model,harmonized_results=results)
    """

    #######################################################################################################################
    """
    print("\nGradient Analysis of local trained model:")
    for name, grad in local_gradients.items():
        print(f"{name}: mean={grad['mean']:.4f}, std={grad['std']:.4f}")
    
    print('\n')

    print("\nGradient Analysis of Global fine-tuned model:")
    for name, grad in global_gradients.items():
        print(f"{name}: mean={grad['mean']:.4f}, std={grad['std']:.4f}")
    print(f"All training done : {silo_name}",end='\n\n')
    """

    """
class SimplifiedAgePredictor(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.main = nn.Sequential(
            nn.Linear(input_size, 512),
            #nn.LayerNorm(512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.02),
            nn.Dropout(DROPOUT_RATE),
            
            nn.Linear(512, 256),
            #nn.LayerNorm(256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.02),
            nn.Dropout(DROPOUT_RATE),
            
            nn.Linear(256, 128),
            #nn.LayerNorm(128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.02),
            
            nn.Linear(128, 1)
        )
"""
"""
for name, param in model.named_parameters():
    if param.requires_grad and param.grad is not None:
        grad = param.grad
        num_elements = grad.numel()
        
        # Handle small gradient tensors
        stats = {
            'mean': grad.mean().item(),
            'max': grad.max().item(),
            'min': grad.min().item(),
            'std': grad.std(unbiased=True).item() if num_elements > 1 else 0.0,
            'data': grad.cpu().numpy().copy()
        }
        # Standardize: (data - mean) / std (avoid division by zero)
        if stats['std'] > 0:
            stats['data'] = ( (stats['data'] - stats['mean']) / stats['std'] ) + 1e-3
        else:
            stats['data'] = stats['data'] - stats['mean'] + 1e-3 # If std=0, just subtract mean
                    

        local_gradients[name] = stats
"""

"""
global_gradients = {}
for name, param in model.named_parameters():
    if param.requires_grad and param.grad is not None:
        grad = param.grad
        num_elements = grad.numel()

        # Compute gradient statistics
        mean = grad.mean().item()
        max_val = grad.max().item()
        min_val = grad.min().item()
        std = grad.std(unbiased=True).item() if num_elements > 1 else 0.0
        
        # Convert to NumPy for standardization
        grad_data = grad.cpu().numpy().copy()

        # Standardize: (data - mean) / std (avoid division by zero)
        if std > 0:
            grad_data = (grad_data - mean) / std + 1e-3
        else:
            grad_data = (grad_data - mean) + 1e-3  # If std=0, just subtract mean
                
        # Store results
        global_gradients[name] = {
            'mean': mean,
            'max': max_val,
            'min': min_val,
            'std': std,
            'data': grad_data  # Standardized gradient
        }
"""
"""
    test_path = train_path.replace('train_data', 'test_data')

    X_test,y_test,_ = preprocess(test_path)
    X_test = scaler.transform(X_test)

    test_loader = DataLoader(TensorDataset(torch.FloatTensor(X_test), torch.FloatTensor(y_test)),batch_size=BATCH_SIZE, shuffle=True)
    
    model.eval()  # Set model to training mode for gradient computation

    # Forward + backward pass to populate gradients
    X_sample, y_sample = next(iter(val_loader))
    X_sample = X_sample.to(DEVICE)
    y_sample = y_sample.to(DEVICE)

    model.zero_grad()
    outputs = model(X_sample).squeeze()
    loss = criterion(outputs, y_sample)
    loss.backward()

    local_gradients = extractGradients(model)
    
    all_actuals = [] # Initialize lists to store actual vs predicted values for MAE calculation
    all_preds = []
    
    # Fine-tuning loop
    total_loss = 0.0
    num_batches = 0
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    # Iterate over global dataset
    for X_batch, y_batch in test_loader:
        X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
        # Ensure target shape
        y_batch = y_batch.view(-1, 1)
        
        # Zero gradients and forward pass
        optimizer.zero_grad()
        predictions = model(X_batch).view(-1,1)
        loss = criterion(predictions, y_batch)
        
        # Backward pass with gradient clipping
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Accumulate loss
        total_loss += loss.item()
        num_batches += 1
        
        # Collect actual and predicted values for MAE calculation
        all_actuals.extend(y_batch.cpu().numpy())
        all_preds.extend(predictions.cpu().detach().numpy())

    # Calculate average loss
    avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')
    print(f'\nTest error of {silo_name} :{avg_loss}\n')
"""