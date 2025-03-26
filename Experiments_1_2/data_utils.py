import os, pandas as pd, numpy as np, torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from config import train_ratio, BATCH_SIZE

def preprocess(data_file_path):
    silo_name = os.path.basename(os.path.dirname(data_file_path))
    data = pd.read_csv(data_file_path)
    
    feature_columns = [str(i) for i in range(1, 1074)]  
    X = data[feature_columns].to_numpy().astype(np.float32)
    y = data['age'].to_numpy().astype(np.float32)
    
    print(f"Silo name : {silo_name}")
    print(f"Feature matrix shape: {X.shape}")
    print(f"Target vector shape: {y.shape}")
    
    return X, y, silo_name

def split_data(X, y, train_ratio=0.7):
    train_size = int(len(X) * train_ratio)
    indices = np.random.permutation(len(X))
    train_idx, val_idx = indices[:train_size], indices[train_size:]
    
    X_train_raw, X_val_raw = X[train_idx].copy(), X[val_idx].copy()
    y_train, y_val = y[train_idx], y[val_idx]
    
    return X_train_raw, y_train, X_val_raw, y_val

def dataloader(X, y, dataset='val'):
    shuffle = dataset == 'train'
    return DataLoader(
        TensorDataset(torch.FloatTensor(X), torch.FloatTensor(y)),
        batch_size=BATCH_SIZE, shuffle=shuffle
    )
