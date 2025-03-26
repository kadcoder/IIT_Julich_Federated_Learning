import torch,os,torch.nn as nn,torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from data_utils import preprocess
from config import DEVICE, BATCH_SIZE, INIT_LR, WEIGHT_DECAY
from model import AgePredictor
import pandas as pd

def evaluate_centralmodel(test_path, model ,scaler):
    X_test, y_test, silo_name = preprocess(test_path)

    #model_path = f"models/{silo_name}_{max_epochs}.pt"
    #model = AgePredictor(X_test.shape[1])
    #model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    #model.to(DEVICE)
    model.eval()
    
    X_test = scaler.transform(X_test)
    test_loader = DataLoader(TensorDataset(torch.FloatTensor(X_test), torch.FloatTensor(y_test)), batch_size=32)
    
    total_loss = 0.0
    criterion = torch.nn.L1Loss()
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            predictions = model(X_batch).view(-1, 1)
            total_loss += criterion(predictions, y_batch.view(-1, 1)).item()
    
    avg_loss = total_loss / len(test_loader)
    print(f"Test error of {silo_name}: {avg_loss}")
    return avg_loss

def evaluate_model(test_path, model, scaler):
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

def predict_age(model, test_path, scaler):
    X_test_raw, y_test, silo_name = preprocess(test_path)
    model.eval()
    X_test = scaler.transform(X_test_raw)
    
    test_loader = DataLoader(TensorDataset(torch.FloatTensor(X_test), torch.FloatTensor(y_test)), batch_size=32, shuffle=False)
    predictions_list = []
    
    with torch.no_grad():
        for X_batch, _ in test_loader:
            X_batch = X_batch.to('cuda')
            predictions = model(X_batch)
            predictions_list.extend(predictions.cpu().numpy().flatten())
    
    results_df = pd.DataFrame({'Predicted_Age': predictions_list})
    results_df.to_csv(f'Predictions_{silo_name}.csv', index=False)
    print(f"Saved predictions to Predictions_{silo_name}.csv")
