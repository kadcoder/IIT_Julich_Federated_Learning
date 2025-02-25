import numpy as np
import pandas as pd
import torch,os
from torch.utils.data import DataLoader, TensorDataset
import config

def dataloader(X, y, dataset='val'or'test'):

    if dataset == 'train':
        data_loader = DataLoader(
            TensorDataset(torch.FloatTensor(X), torch.FloatTensor(y)),
            batch_size=config.BATCH_SIZE, shuffle=True)
    else:
        data_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X), torch.FloatTensor(y)),
        batch_size=config.BATCH_SIZE)

    return data_loader

def split_data(X,y):

    # Compute split index
    train_size = int(len(X) * config.train_ratio)
    indices = np.random.permutation(len(X)) # Shuffle indices

    train_idx, val_idx = indices[:train_size], indices[train_size:]  # Split indices

    # Save raw training data before scaling for gradient computation
    X_train_raw = X[train_idx].copy()
    X_val_raw = X[val_idx].copy()
    y_train = y[train_idx]
    y_val = y[val_idx]

    return X_train_raw,y_train,X_val_raw,y_val

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

