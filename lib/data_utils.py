#%%
import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from typing import Union, Tuple ,Literal, Dict
from torch.utils.data import DataLoader, TensorDataset
from lib.config import BATCH_SIZE
from neuroCombat import neuroCombat
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


def train_combat(silodata_path, silo, global_name):
    #silo = 'CamCAN' #or 'SALD'
    silo_path = os.path.join(silodata_path,f'{silo}/Train_{silo}.csv')
    central_path = os.path.join(silodata_path,f'{global_name}/Train_{global_name}.csv')

    print(f"Loading data from {silo_path} and {central_path}")

    # Example: Load data from two scanners
    # Replace with your actual data paths
    df1 = pd.read_csv(silo_path)
    df2 = pd.read_csv(central_path)  # Shape: (features, subjects)

    # Get feature columns (assuming columns are named '1', '2', ..., '1073')
    feature_columns = [str(i) for i in range(1, 1074)]  # Generates ['1', '2', ..., '1073']
    #print('feature columns',feature_columns)
    # Convert to numpy arrays
    X_1 = np.transpose(df1[feature_columns].to_numpy()).astype(np.float32)  #Features matrix
    y_1 = np.transpose(df1['age'].to_numpy()).astype(np.float32).tolist()            #Target vector

    # Convert to numpy arrays
    X_2 = np.transpose(df2[feature_columns].to_numpy()).astype(np.float32)  #Features matrix
    y_2 = np.transpose(df2['age'].to_numpy()).astype(np.float32).tolist()            #Target vector

    # Combine datasets horizontally (columns = all subjects from both scanners)
    data_combined = np.hstack([X_1, X_2])  # Shape: (features, subjects_total)
    age_list = y_1 + y_2  # Shape: (subjects_total,)

    # Example: 5 subjects from scanner1 and 5 from scanner2
    n_scanner1 = X_1.shape[1]  # Number of subjects in scanner1
    n_scanner2 = X_2.shape[1]  # Number of subjects in scanner2

    covars = {
        'batch': [1]*n_scanner1 + [2]*n_scanner2  # Scanner IDs 
        }
    covars = pd.DataFrame(covars)

    # Harmonize the data
    harmonized_output = neuroCombat(
        dat=data_combined,
        covars=covars,
        batch_col='batch',              # Column name for scanner IDs
        categorical_cols=[],    # Specify categorical variables
        # Optional parameters:
        eb=True,             # Use Empirical Bayes (recommended)
        parametric=True,      # Parametric adjustment (default)
        mean_only=False,      # Adjust both mean and variance (default)
        ref_batch=n_scanner2        # Harmonize to overall average (set to 1 or 2 to use a scanner as reference)
    )

    # Extract harmonized data
    data_harmonized = harmonized_output["data"]
    data_harmonized.shape

    # Split harmonized data back into original scanners (if needed)
    harmonized_scanner1 = data_harmonized[:, :n_scanner1]
    harmonized_scanner2 = data_harmonized[:, n_scanner1:]

    y_1 = np.transpose(df1['age'].to_numpy()).astype(np.float32).tolist()            #Target vector
    y_2 = np.transpose(df2['age'].to_numpy()).astype(np.float32).tolist()            #Target vector

    # Move this line before creating harmonized_combined_data
    feature_columns.append('age')
    #print('feature columns:',feature_columns)

    print("Harmonized data for scanner1:")
    print(harmonized_scanner1.T.shape)
    print(  "Target variable for scanner1:")
    age = np.array(y_1).reshape(-1, 1)
    #print(age.shape)

    # Concatenate the arrays horizontally (adding target as an extra column)
    harmonized_combined_data = np.concatenate((harmonized_scanner1.T, age), axis=1)
    print('harmonized data shape:',harmonized_combined_data.shape)
    harmonized_train = pd.DataFrame(harmonized_combined_data, columns=feature_columns)
    harmonized_path = silo_path.replace('Train','Train_harmonized')
    harmonized_train.to_csv(harmonized_path)
    silo_test_path = silo_path.replace('Train','Test')
    
    return harmonized_output ,silo_test_path

def apply_combat_harmonization(test_path, combat_params, batch_col='batch', categorical_cols=[]):
    """
    Apply pre-trained Combat harmonization to new data.

    Args:
        test_path (str): Path to new test data CSV
        combat_params (dict): Saved Combat parameters from training, containing:
                              - "estimates": Model parameters
                              - "batch_col": Batch column name (e.g., "batch")
                              - "categorical_cols": List of categorical covariates
    """
    # Load test data
    test_data = pd.read_csv(test_path)

    # Clean data (drop rows with missing age)
    test_data_clean = test_data.dropna(subset=['age'], axis=0)

    # Check if test data is empty after cleaning
    if test_data_clean.empty:
        raise ValueError("Test data is empty after dropping rows with missing 'age'.")

    # 1. Add batch column to test_data_clean
    # Assign all test data to a single batch (e.g., batch=1)
    test_data_clean['batch'] = 1  # Replace 1 with your desired batch label

    # 2. Extract covariates (MUST include 'batch' and other training covariates)
    required_covars = ['batch']  # Add other covariates used during training
    covars_test = test_data_clean[required_covars]

    # 3. Extract features (columns '1' to '1073')
    new_data = test_data_clean.loc[:, '1':'1073']

    # Transpose to (features x subjects) format
    new_data_t = new_data.T

    # 4. Apply pre-trained Combat parameters
    # Since neuroCombat doesn't accept 'estimates', remove it from the call:
    harmonized = neuroCombat(
        dat=new_data_t,
        covars=covars_test,
        batch_col=batch_col,
        categorical_cols=categorical_cols,
        # estimates=combat_params["estimates"]  # Remove this line
    )["data"]

    # Convert back to DataFrame and add age labels
    harmonized_df = pd.DataFrame(harmonized.T, columns=new_data.columns)
    harmonized_df['age'] = test_data_clean['age'].values

    return harmonized_df