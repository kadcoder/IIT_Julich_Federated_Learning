import torch
import config
import numpy as np

def compute_gradients(model, criterion, X, y, scaler):
    
    model.eval()
    
    # Ensure input data is on CPU and converted to NumPy
    if isinstance(X, torch.Tensor):
        X = X.cpu().numpy()  # Move to CPU and convert to NumPy
    if isinstance(y, torch.Tensor):
        y = y.cpu().numpy()
    
    # Scale data using sklearn scaler (works with NumPy arrays)
    X_scaled = scaler.transform(X)  # Scale features

    # Convert back to tensors and move to device
    X_tensor = torch.FloatTensor(X_scaled).to(config.DEVICE)
    y_tensor = torch.FloatTensor(y).view(-1, 1).to(config.DEVICE)

    # Ensure parameters require gradients
    for param in model.parameters():
        param.requires_grad_(True)

    # Forward pass
    predictions = model(X_tensor)
    loss = criterion(predictions, y_tensor)

    # Ensure loss requires gradients
    loss.requires_grad_(True)

    # Compute gradients
    model.zero_grad()
    loss.backward()

    # Extract gradients from model parameters
    grads = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            grads[name] = param.grad.clone().detach().cpu().numpy().astype(np.float32)
        #else:
        #    grads[name] = np.zeros_like(param.cpu().detach().numpy(), dtype=np.float32)  # Handle None gradients

    return grads