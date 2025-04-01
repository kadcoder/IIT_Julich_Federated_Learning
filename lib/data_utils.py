import os, pandas as pd, numpy as np, torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from config import BATCH_SIZE


# TODO: Add docstrings to the functions
# TODO: Add type hints to the functions
# TODO: change the name to silo_data_loading
def preprocess(data_file: str | pd.DataFrame) -> tuple[np.ndarray,
                                                       np.ndarray,
                                                       str]:
    """
    Preprocess the data by loading it from a file or DataFrame,
    extracting features and target variables, and scaling the features.
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

    # TODO: Remove the age for the data to get X
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
