#%%
import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from typing import Union, Tuple ,Literal, Dict
from torch.utils.data import DataLoader, TensorDataset
from lib.config import BATCH_SIZE
#%%

def preprocess(data_file: Union[str, pd.DataFrame]) -> Tuple[np.ndarray, np.ndarray, str]:
    """
    Preprocess the data by loading it from a file or DataFrame,
    extracting features and target variables, and scaling the features.
    
    Parameters:
        data_file (Union[str, pd.DataFrame]):  Specify whether it can be a path or a DataFrame
        
    Returns:
        Tuple[np.ndarray, np.ndarray, str]: 
            - X: Feature matrix of shape (n_samples, n_features)
            - y: Target vector (age)
            - silo_name: Name of the silo (if applicable)
    """
    
    # Check if data_file is a string (file path) or a DataFrame
    if isinstance(data_file, str):
        if not os.path.exists(data_file):
            raise FileNotFoundError(f"File not found: {data_file}")
        silo_name = os.path.basename(os.path.dirname(data_file))
        data = pd.read_csv(data_file)
    elif isinstance(data_file, pd.DataFrame):
        data = data_file
        silo_name = ''
    else:
        raise TypeError("data_file should be a file path (str) or a pandas DataFrame")  # noqa

    # TODO: Remove the 'age' column from the data to get feature matrix X
    feature_columns = [str(i) for i in range(1, 1074)]
    X = data[feature_columns].to_numpy().astype(np.float32)
    y = data['age'].to_numpy().astype(np.float32)

    #print(f"Silo name : {silo_name}")
    #print(f"Feature matrix shape: {X.shape}")
    #print(f"Target vector shape: {y.shape}")

    return X, y, silo_name

def dataloader(
    X: np.ndarray,
    y: np.ndarray,
    dataset: Literal['train', 'val'] = 'val'
) -> DataLoader:
    
    """
    Creates a PyTorch DataLoader from input feature and target arrays.

    Parameters:
    -----------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features).

    y : np.ndarray
        Target array of shape (n_samples,) or (n_samples, 1).

    dataset : Literal['train', 'val'], optional (default='val')
        Specifies the type of dataset. If 'train', the DataLoader will shuffle the data.

    Returns:
    --------
    DataLoader
        A PyTorch DataLoader that wraps the dataset for iteration during training or validation.
    """
    shuffle = dataset == 'train'

    return DataLoader(
        TensorDataset(torch.FloatTensor(X), torch.FloatTensor(y)),
        batch_size=BATCH_SIZE,
        shuffle=shuffle
    )

def compute_gradients(
    model: nn.Module,
    criterion: nn.Module,
    X_scaled: torch.Tensor,
    y: torch.Tensor,
    total_samples: int
) -> Dict[str, torch.Tensor]:
    """
    Computes the scaled gradients of a model with respect to input data and loss.

    Parameters
    ----------
    model : nn.Module
        The neural network model whose gradients are to be computed.
    criterion : nn.Module
        The loss function used to compute the backward gradients.
    X_scaled : torch.Tensor
        Scaled feature input (e.g., normalized or standardized).
    y : torch.Tensor
        Ground truth target values.
    total_samples : int
        Total number of samples across all silos (used for gradient scaling).

    Returns
    -------
    grads : Dict[str, torch.Tensor]
        Dictionary mapping parameter names to their computed and scaled gradients.
    """

    model.train()
    X_tensor = torch.FloatTensor(X_scaled).to('cuda')
    y_tensor = torch.FloatTensor(y).view(-1, 1).to('cuda')

    for param in model.parameters():
        param.requires_grad_(True)

    predictions = model(X_tensor)
    loss = criterion(predictions, y_tensor)
    loss.requires_grad_(True)

    model.zero_grad()
    loss.backward()

    grads: Dict[str, torch.Tensor] = {
        name: param.grad.clone().detach() * (X_scaled.shape[0] / total_samples)
        for name, param in model.named_parameters()
        if param.grad is not None
    }

    return grads
