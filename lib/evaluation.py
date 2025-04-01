import torch
import os
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from data_utils import preprocess
from config import DEVICE, BATCH_SIZE, INIT_LR, WEIGHT_DECAY
from model import AgePredictor
import pandas as pd


def evaluate_centralmodel(test_path, best_model, scaler):
    # TODO: Add docstrings to the functions
    # TODO: Add type hints to the functions
    # load the test data
    X_test, y_test, silo_name = preprocess(test_path)

    best_model.eval()
    best_model.to(DEVICE)
    X_test = scaler.transform(X_test)
    test_loader = DataLoader(TensorDataset(torch.FloatTensor(X_test),
                                           torch.FloatTensor(y_test)),
                             batch_size=32)

    total_loss = 0.0
    criterion = torch.nn.L1Loss()
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            predictions = best_model(X_batch).view(-1, 1)
            total_loss += criterion(predictions, y_batch.view(-1, 1)).item()

    avg_loss = total_loss / len(test_loader)
    return avg_loss


def evaluate_model(test_path, model, scaler):
    # TODO: Add docstrings to the functions
    # TODO: Add type hints to the functions
    X_test_raw, y_test, silo_name = preprocess(test_path)
    model.eval()
    X_test = scaler.transform(X_test_raw)
    
    test_loader = DataLoader(TensorDataset(torch.FloatTensor(X_test), torch.FloatTensor(y_test)), batch_size=32, shuffle=False)
    criterion = torch.nn.L1Loss()
    total_loss = 0.0
    
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to('cuda'), y_batch.to('cuda')
            predictions = model(X_batch)
            total_loss += criterion(predictions.view(-1, 1), y_batch.view(-1, 1)).item()
    
    return total_loss / len(test_loader)

def predict_age(model, test_path, scaler, epochs):
    X_test_raw, y_test, silo_name = preprocess(test_path)
    model.eval()
    X_test = scaler.transform(X_test_raw)

    test_loader = DataLoader(TensorDataset(torch.FloatTensor(X_test), torch.FloatTensor(y_test)), batch_size=32, shuffle=False)
    predictions_list = []
    actual_age_list = []
    
    with torch.no_grad():
        for X_batch, Y_batch in test_loader:
            X_batch = X_batch.to('cuda')
            predictions = model(X_batch)
            predictions_list.extend(predictions.cpu().numpy().flatten())
            actual_age_list.extend(Y_batch.numpy().flatten())
    
    results_df = pd.DataFrame({'Actual_Age': actual_age_list, 'Predicted_Age': predictions_list})
    results_df.to_csv(f'Predictions_{silo_name}_{epochs}.csv', index=False)
    print(f"Saved predictions to Predictions_{silo_name}_{epochs}.csv")
    
    # Plot predicted vs actual age
    plt.figure(figsize=(8, 6))
    plt.scatter(actual_age_list, predictions_list, alpha=0.5, label='Predictions')
    plt.plot([min(actual_age_list), max(actual_age_list)], [min(actual_age_list), max(actual_age_list)], 'r--', label='45° Line')
    plt.xlabel('Actual Age')
    plt.ylabel('Predicted Age')
    plt.title(f'Predicted vs Actual Age ({silo_name})')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'Predicted_vs_Actual_Age_{silo_name}_{epochs}.png')

