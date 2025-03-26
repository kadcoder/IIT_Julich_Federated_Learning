import torch
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd

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
