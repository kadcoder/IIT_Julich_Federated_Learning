# %%
import torch
import os
import pandas as pd
import torch.nn as nn
import matplotlib.pyplot as plt
from typing import Tuple
import copy
from torch.utils.data import DataLoader, TensorDataset

from lib.data_utils import preprocess
from lib.config import DEVICE
# %%


def evaluate_centralmodel(test_path: str, best_model: nn.Module, scaler) -> float:
    """
    Evaluates a centrally trained model on a given test dataset.

    Args:
        test_path (str): Path to the test dataset.
        best_model (nn.Module): Trained PyTorch model to evaluate.
        scaler: Scaler object used for feature normalization.

    Returns:
        float: Average mean absolute error (MAE) loss on the test set.
    """
    X_test, y_test, silo_name = preprocess(test_path)

    best_model.eval()
    best_model.to(DEVICE)
    X_test = scaler.transform(X_test)

    test_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_test), torch.FloatTensor(y_test)),
        batch_size=32,
    )

    total_loss = 0.0
    criterion = nn.L1Loss()

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            predictions = best_model(X_batch).view(-1, 1)
            total_loss += criterion(predictions, y_batch.view(-1, 1)).item()

    avg_loss = total_loss / len(test_loader)
    return avg_loss


def evaluate_model(test_path: str, model: nn.Module, scaler) -> float:
    """
    Evaluates a model (e.g., from a silo) on its respective test set.

    Args:
        test_path (str): Path to the test dataset.
        model (nn.Module): Trained PyTorch model.
        scaler: Scaler used for feature normalization.

    Returns:
        float: Average mean absolute error (MAE) loss on the test set.
    """
    X_test_raw, y_test, silo_name = preprocess(test_path)

    model.eval()
    X_test = scaler.transform(X_test_raw)

    test_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_test), torch.FloatTensor(y_test)),
        batch_size=32,
        shuffle=False,
    )

    criterion = nn.L1Loss()
    total_loss = 0.0

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to("cuda"), y_batch.to("cuda")
            predictions = model(X_batch)
            total_loss += criterion(predictions.view(-1, 1), y_batch.view(-1, 1)).item()

    return total_loss / len(test_loader)


def predict_age(model: nn.Module, test_path: str, result_path: str, scaler) -> None:
    """
    Predicts age using a trained model and saves the predictions and a plot.

    Args:
        model (nn.Module): Trained PyTorch model.
        test_path (str): Path to the test dataset.
        result_path (str): Directory where prediction results and plots will be saved.
        scaler: Scaler used for feature normalization.

    Returns:
        None
    """
    X_test_raw, y_test, silo_name = preprocess(test_path)

    model.eval()
    X_test = scaler.transform(X_test_raw)

    test_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_test), torch.FloatTensor(y_test)),
        batch_size=32,
        shuffle=False,
    )

    predictions_list = []
    actual_age_list = []

    with torch.no_grad():
        for X_batch, Y_batch in test_loader:
            X_batch = X_batch.to("cuda")
            predictions = model(X_batch)
            predictions_list.extend(predictions.cpu().numpy().flatten())
            actual_age_list.extend(Y_batch.numpy().flatten())

    results_df = pd.DataFrame(
        {"Actual_Age": actual_age_list, "Predicted_Age": predictions_list}
    )

    result_age_path = os.path.join(result_path, f"Predictions_{silo_name}.csv")
    os.makedirs(os.path.dirname(result_age_path), exist_ok=True)
    results_df.to_csv(result_age_path, index=False)
    print(f"Saved predictions to {result_age_path}")

    # Plot predicted vs actual age
    plt.figure(figsize=(8, 6))
    plt.scatter(actual_age_list, predictions_list, alpha=0.5, label="Predictions")
    plt.plot(
        [min(actual_age_list), max(actual_age_list)],
        [min(actual_age_list), max(actual_age_list)],
        "r--",
        label="45° Line",
    )
    plt.xlabel("Actual Age")
    plt.ylabel("Predicted Age")
    plt.title(f"Predicted vs Actual Age ({silo_name})")
    plt.legend()
    plt.grid(True)

    plot_path = os.path.join(result_path, f"Predicted_vs_Actual_Age_{silo_name}.png")
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    plt.savefig(plot_path)
    plt.close()


def compute_validation_loss(
    gmodel: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    current_best_loss: float,
    current_best_model: nn.Module,
    device: torch.device = DEVICE,
) -> Tuple[float, nn.Module]:
    """
    Calculate validation loss and return the best model (either current or previous best)

    Parameters
    ----------
    gmodel : nn.Module
        Current global model to evaluate
    val_loader : DataLoader
        Validation dataloader
    criterion : nn.Module
        Loss criterion
    current_best_loss : float
        Best validation loss from previous epochs
    current_best_model : nn.Module
        Best model from previous epochs
    device : torch.device
        Device for computation

    Returns
    -------
    Tuple[float, nn.Module]
        Updated best validation loss and corresponding model
    """
    gmodel.eval()
    total_loss = 0.0

    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            y_batch = y_batch.view(-1, 1)

            outputs = gmodel(X_batch).view(-1, 1)
            loss = criterion(outputs, y_batch)
            total_loss += loss.item()

    val_loss = total_loss / len(val_loader)

    # Compare with current best and return appropriate model
    if val_loss < current_best_loss:
        print(
            f"New best validation loss: {val_loss:.4f} (improved from {current_best_loss:.4f})"
        )
        best_global_loss = val_loss
        best_model_global = copy.deepcopy(gmodel)
    else:
        best_model_global = current_best_model
        best_global_loss = current_best_loss
        print(
            f"Validation loss: {val_loss:.4f} (no improvement from {current_best_loss:.4f})"
        )

    return best_global_loss, best_model_global
