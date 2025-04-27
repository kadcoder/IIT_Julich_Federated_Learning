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


def train_combat(
    silodata_path: str,
    silo_name: str,
    global_name: str
) -> Tuple[Dict[str, np.ndarray], str]:
    """
    Trains a ComBat harmonization model by combining local silo data with global data
    and applies the trained ComBat parameters to the silo's training set.

    Parameters:
    -----------
    silodata_path : str
        Path to the directory containing silo and global CSV files.

    silo_name : str
        Name of the local silo (e.g., 'CamCAN', 'SALD').

    global_name : str
        Name of the global/central dataset.

    Returns:
    --------
    harmonized_output : dict
        Output dictionary from neuroCombat containing harmonized data and model parameters.

    silo_test_path : str
        Path to the corresponding test data CSV for the given silo.
    """

    # Construct paths
    silo_train_path = os.path.join(silodata_path, f'{silo_name}/Train_{silo_name}.csv')
    global_train_path = os.path.join(silodata_path, f'{global_name}/Train_{global_name}.csv')

    print(f"Loading data from {silo_train_path} and {global_train_path}")

    # Load training data
    df_silo = pd.read_csv(silo_train_path)
    df_global = pd.read_csv(global_train_path)

    # Extract feature columns
    feature_columns = [str(i) for i in range(1, 1074)]  # ['1', '2', ..., '1073']

    # Prepare feature matrices and target vectors
    X_silo = np.transpose(df_silo[feature_columns].to_numpy()).astype(np.float32)
    y_silo = df_silo['age'].to_numpy().astype(np.float32).tolist()

    X_global = np.transpose(df_global[feature_columns].to_numpy()).astype(np.float32)
    y_global = df_global['age'].to_numpy().astype(np.float32).tolist()

    # Combine silo and global data
    data_combined = np.hstack([X_silo, X_global])  # (features, total subjects)
    age_list = y_silo + y_global

    # Define batch covariates
    n_silo = X_silo.shape[1]
    n_global = X_global.shape[1]

    covars = pd.DataFrame({
        'batch': [1]*n_silo + [2]*n_global,
        'age': age_list
    })

    # Apply neuroCombat harmonization
    harmonized_output = neuroCombat(
        dat=data_combined,
        covars=covars,
        batch_col='batch',
        categorical_cols=[],
        eb=True,
        parametric=True,
        mean_only=False,
        ref_batch=None
    )

    # Extract harmonized data
    data_harmonized = harmonized_output["data"]

    # Split back into silo and global parts
    harmonized_silo = data_harmonized[:, :n_silo]
    harmonized_global = data_harmonized[:, n_silo:]

    # Prepare harmonized DataFrame for the silo
    age_array = np.array(y_silo).reshape(-1, 1)
    harmonized_combined = np.concatenate((harmonized_silo.T, age_array), axis=1)

    # Add 'age' column to feature list
    feature_columns_with_age = feature_columns + ['age']

    harmonized_silo_df = pd.DataFrame(harmonized_combined, columns=feature_columns_with_age)

    # Save harmonized training data
    harmonized_train_path = silo_train_path.replace('Train', 'Train_harmonized')
    harmonized_silo_df.to_csv(harmonized_train_path, index=False)

    # Derive silo test data path
    silo_test_path = silo_train_path.replace('Train', 'Test')

    print(f"Harmonized training data saved to {harmonized_train_path}")
    print(f"Corresponding silo test path: {silo_test_path}")

    return harmonized_output, silo_test_path

def apply_combat_harmonization(test_path, combat_params, batch_col='batch', categorical_cols=[]):
    """
    Apply pre-trained ComBat harmonization parameters to new test data.

    This function takes unseen test data and harmonizes it using previously learned ComBat parameters.
    It assumes the test data includes an 'age' column and features labeled '1' through '1073'.
    The batch label is assigned as 1 by default.

    Parameters
    ----------
    test_path : str
        Path to the CSV file containing the new test dataset to be harmonized.
    combat_params : dict
        Dictionary containing the pre-trained ComBat harmonization parameters, 
        typically obtained from a prior training phase. Must contain:
            - "estimates": Estimated model parameters.
            - "batch_col": Name of the batch column (e.g., 'batch').
            - "categorical_cols": List of categorical covariates.
    batch_col : str, optional
        The name of the column identifying batch (scanner/site) information (default is 'batch').
    categorical_cols : list, optional
        List of column names in the covariates that should be treated as categorical variables 
        (default is an empty list).

    Returns
    -------
    harmonized_df : pandas.DataFrame
        DataFrame containing the harmonized feature data along with the 'age' column. 
        The features are harmonized based on the provided pre-trained ComBat parameters.

    Raises
    ------
    ValueError
        If the test dataset becomes empty after dropping rows with missing 'age' values.

    Example
    -------
    >>> harmonized_test_df = apply_combat_harmonization(
            test_path='/path/to/Test_Silo.csv',
            combat_params=trained_combat_output,
            batch_col='batch',
            categorical_cols=[]
        )
    """
    # Load test data
    test_data = pd.read_csv(test_path)

    # Clean data (drop rows with missing age)
    test_data_clean = test_data.dropna(subset=['age'], axis=0)

    if test_data_clean.empty:
        raise ValueError("Test data is empty after dropping rows with missing 'age'.")

    # Assign batch ID (assume test data is from silo with batch=1)
    test_data_clean['batch'] = 1

    # Extract covariates
    required_covars = ['age', 'batch']
    covars_test = test_data_clean[required_covars]

    # Extract feature data
    new_data = test_data_clean.loc[:, '1':'1073']
    new_data_t = new_data.T  # Transpose: features x subjects

    # Apply pre-trained Combat harmonization
    harmonized = neuroCombat(
        dat=new_data_t,
        covars=covars_test,
        batch_col=batch_col,
        categorical_cols=categorical_cols
        #estimates=combat_params["estimates"]  # Use saved parameters
    )["data"]

    # Convert harmonized data back to DataFrame
    harmonized_df = pd.DataFrame(harmonized.T, columns=new_data.columns)
    harmonized_df['age'] = test_data_clean['age'].values

    return harmonized_df
