import torch
from sklearn.preprocessing import StandardScaler

import config
import model as m
import grads
from brain_age_parcels.utils import utils

def train_localmodel(train_path,global_path,epochs):

    X, y,silo_name = utils.preprocess(train_path)

    X_train_raw,y_train,X_val_raw,y_val = utils.split_data(X,y)
    
    scaler = StandardScaler()  # Normalization
    X_train = scaler.fit_transform(X_train_raw)
    X_val = scaler.transform(X_val_raw)

    print(f"train dataset shape: {X_train.shape}")  # Should be (n_samples, 1073)
    print(f"validation dataset shape: {X_val.shape}")   # Should be (n_samples, 1073)
    
    # Convert to tensors
    train_loader = utils.dataloader(X_train,y_train,'train')
    val_loader = utils.dataloader(X_val,y_val)

    model = m.AgePredictor(X_train.shape[1]).to(config.DEVICE)
    model.load_state_dict(torch.load(global_path, map_location=config.DEVICE))

    criterion = torch.nn.L1Loss()

    #optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9, weight_decay=WEIGHT_DECAY)
    optimizer = torch.optim.Adam(model.parameters(), lr = config.INIT_LR,weight_decay=config.WEIGHT_DECAY)

    # Here, ReduceLROnPlateau monitors the validation loss and reduces the LR by a factor of 0.5 if no improvement is seen for 2 epochs.
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)


    train_losses, val_losses = [], []

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(config.DEVICE), y_batch.to(config.DEVICE)
        
            # Ensure target shape is (batch_size, 1)
            y_batch = y_batch.view(-1, 1)
            optimizer.zero_grad()
            predictions = model(X_batch)

            # Ensure predictions match target shape
            predictions = predictions.view(-1, 1)

            loss = criterion(predictions, y_batch)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        train_loss = total_train_loss / len(train_loader)
        train_losses.append(train_loss)

        # Validation
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for X_val, y_val in val_loader:
                X_val, y_val = X_val.to(config.DEVICE), y_val.to(config.DEVICE)
                # Ensure target shape
                y_val = y_val.view(-1, 1)

                val_predictions = model(X_val)
                val_predictions = val_predictions.view(-1, 1)

                loss = criterion(val_predictions, y_val)
                total_val_loss += loss.item()

        val_loss = total_val_loss / len(val_loader)
        val_losses.append(val_loss)

        # Step the scheduler with the validation loss.
        scheduler.step(val_loss)

        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
    
    # Compute gradients on local training data
    local_gradients = grads.compute_gradients(model, criterion, X_train_raw, y_train, scaler)

    # Save the final model weights
    torch.save(model.state_dict(), f"{silo_name}.pt")
    print(f"Model weights saved as {silo_name}.pt")

    return local_gradients, scaler