
import torch.nn as nn
import torch
import os
from sklearn.preprocessing import StandardScaler
import model as m
import config
from brain_age_parcels.utils import utils
import grads

GLOBAL_SCALER = None

def train_globalmodel(global_train_path, silo_name):

    X_global, y_global, _ = utils.preprocess(global_train_path)
    global GLOBAL_SCALER

    # Use a new scaler fit on global data (if distribution differs)
    if GLOBAL_SCALER is None:
        GLOBAL_SCALER = StandardScaler().fit(X_global)

    X_global_scaled = GLOBAL_SCALER.transform(X_global)
    
    # Load local model
    model = m.AgePredictor(X_global.shape[1]).to(config.DEVICE)
    model.load_state_dict(torch.load(f"{silo_name}.pt", map_location=config.DEVICE))
    model.train()  # Ensure training mode
    
    # New optimizer for global training
    optimizer = torch.optim.Adam(model.parameters(), lr = config.INIT_LR, weight_decay=config.WEIGHT_DECAY)
    #optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9,weight_decay=WEIGHT_DECAY)
    criterion = torch.nn.L1Loss()
    # This scheduler reduces the learning rate by a factor of 0.5 if the average loss does not improve for 2 epochs.
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)
    
    # Fine-tuning on global data
    global_loader = utils.dataloader(X_global_scaled,y_global)
    
    epoch_loss = 0.0

    for epoch in range(config.FINE_TUNE_EPOCHS):
        all_actuals = []  # Initialize lists to store actual vs predicted values for MAE calculation
        all_preds = []
        total_loss = 0.0
        num_batches = 0
        
        print(f"Epoch {epoch+1}/{config.FINE_TUNE_EPOCHS}")
        
        # Iterate over global dataset
        for X_batch, y_batch in global_loader:
            X_batch, y_batch = X_batch.to(config.DEVICE), y_batch.to(config.DEVICE)
            
            # Ensure target shape
            y_batch = y_batch.view(-1, 1)
            
            # Zero gradients and forward pass
            optimizer.zero_grad()
            predictions = model(X_batch).view(-1, 1)
            loss = criterion(predictions, y_batch)
            
            # Backward pass with gradient clipping
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # Accumulate loss
            total_loss += loss.item()
            num_batches += 1
            
            # Collect actual and predicted values for MAE calculation
            all_actuals.extend(y_batch.cpu().numpy())
            all_preds.extend(predictions.cpu().detach().numpy())
        
        # Calculate average loss for the epoch
        avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')
        epoch_loss +=avg_loss
        print(f'Average Test Error of {silo_name} in Epoch {epoch+1}: {avg_loss}\n')
        # Step the scheduler with the current average loss
        scheduler.step(avg_loss)
    
    # Compute gradients on global data
    global_gradients = grads.compute_gradients(model, criterion, X_global, y_global, GLOBAL_SCALER)
    print(f'Avg loss on global data : {epoch_loss/config.FINE_TUNE_EPOCHS}')

    if os.path.exists(f'{silo_name}.pt'):
        os.remove(f'{silo_name}.pt')
        print(f"model Deleted: {silo_name}")

    return global_gradients