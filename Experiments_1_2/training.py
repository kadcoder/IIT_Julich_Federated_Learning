import torch, torch.optim as optim, copy,os
from sklearn.preprocessing import StandardScaler
from model import AgePredictor
from data_utils import preprocess, split_data, dataloader
from config import INIT_LR, DEVICE, WEIGHT_DECAY, GRADIENT_CLIP,GLOBAL_SCALER
from harmonize import harmonize_localglobal

def compute_gradients(model, criterion, X, y, scaler, total):
    model.train()
    X_scaled = scaler.transform(X)  
    X_tensor = torch.FloatTensor(X_scaled).to('cuda')
    y_tensor = torch.FloatTensor(y).view(-1, 1).to('cuda')
    
    for param in model.parameters():
        param.requires_grad_(True)
    
    predictions = model(X_tensor)
    loss = criterion(predictions, y_tensor)
    loss.requires_grad_(True)
    model.zero_grad()
    loss.backward()
    
    grads = {name: param.grad.clone().detach() * (X.shape[0] / total) 
             for name, param in model.named_parameters() if param.grad is not None}
    
    return grads

def train_localmodelgrads(train_path, model, total_samples, epochs=15):

    X, y,silo = preprocess(train_path)

    lmodel = copy.deepcopy(model)

    X_train_raw,y_train,X_val_raw,y_val = split_data(X,y)
    epoch_lgrads = {}
    
    scaler = StandardScaler()  # Normalization
    X_train = scaler.fit_transform(X_train_raw)
    X_val = scaler.transform(X_val_raw)

    print(f"train dataset shape: {X_train.shape}")  # Should be (n_samples, 1073)
    print(f"validation dataset shape: {X_val.shape}")   # Should be (n_samples, 1073)
    
    # Convert to tensors
    train_loader = dataloader(X_train,y_train,'train')
    val_loader = dataloader(X_val,y_val)
    
    criterion = torch.nn.L1Loss()

    # Modified optimizer configuration
    optimizer = torch.optim.SGD( lmodel.parameters(), lr=INIT_LR, momentum=0.9,  weight_decay=WEIGHT_DECAY )

    # Modified learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR( optimizer, T_max=epochs,  eta_min=INIT_LR/100 )

    # Add gradient clipping
    torch.nn.utils.clip_grad_norm_(lmodel.parameters(), max_norm=1.0)

    # Add alignment loss with global model
    alignment_weight = 0.3  # Controls global-local alignment strength
    
    global_model = copy.deepcopy(model).eval()  # Use current global model
    train_losses,val_losses = [],[]

    # Training loop modifications
    for epoch in range(epochs):
        lmodel.train()
        total_train_loss = 0
        total_alignment_loss = 0  # <-- ADDED

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            y_batch = y_batch.view(-1, 1)
            
            optimizer.zero_grad()
            predictions = lmodel(X_batch).view(-1, 1)
            
            # Original loss
            loss = criterion(predictions, y_batch)
            
            # Alignment loss with global model
            alignment_loss = 0  # <-- ADDED
            for (l_param, g_param) in zip(lmodel.parameters(), global_model.parameters()):
                alignment_loss += torch.norm(l_param - g_param.detach(), p=2)
            
            total_loss = loss + alignment_weight * alignment_loss  # <-- ADDED
            total_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(lmodel.parameters(), GRADIENT_CLIP)  # <-- ADDED
            
            optimizer.step()
            
            total_train_loss += loss.item()
            total_alignment_loss += alignment_loss.item()  # <-- ADDED

        # Changed scheduler step
        scheduler.step()  # <-- CHANGED
        train_loss = total_train_loss / len(train_loader)
        train_losses.append(train_loss)
        epoch_lgrads[f'{epoch+1}'] = compute_gradients(lmodel, criterion, X_train_raw, y_train, scaler, total_samples)

        # Validation
        lmodel.eval()
        total_val_loss = 0
        with torch.no_grad():
            for X_val, y_val in val_loader:
                X_val, y_val = X_val.to(DEVICE), y_val.to(DEVICE)
                # Ensure target shape
                y_val = y_val.view(-1, 1)

                val_predictions = lmodel(X_val)
                val_predictions = val_predictions.view(-1, 1)

                loss = criterion(val_predictions, y_val)
                total_val_loss += loss.item()

        val_loss = total_val_loss / len(val_loader)
        val_losses.append(val_loss)

        #print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
    
    # Compute gradients on local training data
    local_gradients = compute_gradients(lmodel, criterion, X_train_raw, y_train, scaler, total_samples)
    avg_train_loss = sum(train_losses)/ len(train_losses)
    avg_val_loss = sum(val_losses)/len(val_losses)

    print(f'Train loss :{avg_train_loss} and Val loss :{avg_val_loss} of silo name :{silo}')

    return epoch_lgrads, train_losses, val_losses, scaler

def train_localmodel(train_path, model, total_samples, epochs=15):

    X, y,silo = preprocess(train_path)

    lmodel = copy.deepcopy(model)

    X_train_raw,y_train,X_val_raw,y_val = split_data(X,y)
    
    scaler = StandardScaler()  # Normalization
    X_train = scaler.fit_transform(X_train_raw)
    X_val = scaler.transform(X_val_raw)

    print(f"train dataset shape: {X_train.shape}")  # Should be (n_samples, 1073)
    print(f"validation dataset shape: {X_val.shape}")   # Should be (n_samples, 1073)
    
    # Convert to tensors
    train_loader = dataloader(X_train,y_train,'train')
    val_loader = dataloader(X_val,y_val)
    
    criterion = torch.nn.L1Loss()

    # Modified optimizer configuration
    optimizer = torch.optim.SGD( lmodel.parameters(), lr=INIT_LR, momentum=0.9,  weight_decay=WEIGHT_DECAY )

    # Modified learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR( optimizer, T_max=epochs,  eta_min=INIT_LR/100 )

    # Add gradient clipping
    torch.nn.utils.clip_grad_norm_(lmodel.parameters(), max_norm=1.0)

    # Add alignment loss with global model
    alignment_weight = 0.3  # Controls global-local alignment strength
    
    global_model = copy.deepcopy(model).eval()  # Use current global model
    train_losses,val_losses = [],[]

    # Training loop modifications
    for epoch in range(epochs):
        lmodel.train()
        total_train_loss = 0
        total_alignment_loss = 0  # <-- ADDED

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            y_batch = y_batch.view(-1, 1)
            
            optimizer.zero_grad()
            predictions = lmodel(X_batch).view(-1, 1)
            
            # Original loss
            loss = criterion(predictions, y_batch)
            
            # Alignment loss with global model
            alignment_loss = 0  # <-- ADDED
            for (l_param, g_param) in zip(lmodel.parameters(), global_model.parameters()):
                alignment_loss += torch.norm(l_param - g_param.detach(), p=2)
            
            total_loss = loss + alignment_weight * alignment_loss  # <-- ADDED
            total_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(lmodel.parameters(), GRADIENT_CLIP)  # <-- ADDED
            
            optimizer.step()
            
            total_train_loss += loss.item()
            total_alignment_loss += alignment_loss.item()  # <-- ADDED

        # Changed scheduler step
        scheduler.step()  # <-- CHANGED
        train_loss = total_train_loss / len(train_loader)
        train_losses.append(train_loss)

        # Validation
        lmodel.eval()
        total_val_loss = 0
        with torch.no_grad():
            for X_val, y_val in val_loader:
                X_val, y_val = X_val.to(DEVICE), y_val.to(DEVICE)
                # Ensure target shape
                y_val = y_val.view(-1, 1)

                val_predictions = lmodel(X_val)
                val_predictions = val_predictions.view(-1, 1)

                loss = criterion(val_predictions, y_val)
                total_val_loss += loss.item()

        val_loss = total_val_loss / len(val_loader)
        val_losses.append(val_loss)

        #print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
    
    # Compute gradients on local training data
    local_gradients = compute_gradients(lmodel, criterion, X_train_raw, y_train, scaler, total_samples)
    avg_train_loss = sum(train_losses)/ len(train_losses)
    avg_val_loss = sum(val_losses)/len(val_losses)

    print(f'Train loss :{avg_train_loss} and Val loss :{avg_val_loss} of silo name :{silo}')

    return local_gradients, train_losses, val_losses, scaler

def train_globalmodel(gmodel,global_train_path,total_samples,FINE_TUNE_EPOCHS=10):

    #global_train_path = os.path.join(silodata,f'eNki/Train_eNki.csv')
    X_global, y_global, silo = preprocess(global_train_path)

    X_train_raw,y_train,X_val_raw,y_val = split_data(X_global,y_global)
    
    global GLOBAL_SCALER
    GLOBAL_SCALER = None

    # Use a new scaler fit on global data (if distribution differs)
    if GLOBAL_SCALER is None:
        GLOBAL_SCALER = StandardScaler()

    X_train = GLOBAL_SCALER.fit_transform(X_train_raw)
    X_val = GLOBAL_SCALER.transform(X_val_raw)

    print(f"train dataset shape: {X_train.shape}")  # Should be (n_samples, 1073)
    print(f"validation dataset shape: {X_val.shape}")   # Should be (n_samples, 1073)
    
    # Convert to tensors
    train_loader = dataloader(X_train,y_train,'train')
    val_loader = dataloader(X_val,y_val)

    optimizer = optim.SGD(gmodel.parameters(), lr=INIT_LR, momentum=0.9, weight_decay=WEIGHT_DECAY)
    criterion = torch.nn.L1Loss()

    # This scheduler reduces the learning rate by a factor of 0.5 if the average loss does not improve for 2 epochs.
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)
    
    train_losses, val_losses = [], []

    for epoch in range(FINE_TUNE_EPOCHS):
        gmodel.train()
        total_train_loss = 0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
        
            # Ensure target shape is (batch_size, 1)
            y_batch = y_batch.view(-1, 1)
            optimizer.zero_grad()
            predictions = gmodel(X_batch)

            # Ensure predictions match target shape
            predictions = predictions.view(-1, 1)

            loss = criterion(predictions, y_batch)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        train_loss = total_train_loss / len(train_loader)
        train_losses.append(train_loss)

        # Validation
        gmodel.eval()
        total_val_loss = 0
        with torch.no_grad():
            for X_val, y_val in val_loader:
                X_val, y_val = X_val.to(DEVICE), y_val.to(DEVICE)
                # Ensure target shape
                y_val = y_val.view(-1, 1)

                val_predictions = gmodel(X_val)
                val_predictions = val_predictions.view(-1, 1)

                loss = criterion(val_predictions, y_val)
                total_val_loss += loss.item()

        val_loss = total_val_loss / len(val_loader)
        val_losses.append(val_loss)

        # Step the scheduler with the validation loss.
        scheduler.step(val_loss)

        print(f"Epoch {epoch+1}/{FINE_TUNE_EPOCHS} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
    
    # Compute gradients on global data
    global_gradients = compute_gradients(gmodel, criterion, X_global, y_global, GLOBAL_SCALER, total_samples)
    avg_train_loss = sum(train_losses)/len(train_losses)
    avg_val_loss = sum(val_losses)/len(val_losses)
    print(f'Avg loss on global data of {silo}: {sum(train_losses)/len(train_losses)}')

    return gmodel, avg_val_loss, train_losses, val_losses,GLOBAL_SCALER

def train_localglobal(train_path,model,total_samples,epochs=15):

    X, y,silo = preprocess(train_path)

    lmodel = copy.deepcopy(model)

    X_train_raw,y_train,X_val_raw,y_val = split_data(X,y)

    epoch_lgrads,epoch_ggrads = {},{}
    
    scaler = StandardScaler()  # Normalization
    X_train = scaler.fit_transform(X_train_raw)
    X_val = scaler.transform(X_val_raw)

    print(f"train dataset shape: {X_train.shape}")  # Should be (n_samples, 1073)
    print(f"validation dataset shape: {X_val.shape}")   # Should be (n_samples, 1073)
    
    # Convert to tensors
    train_loader = dataloader(X_train,y_train,'train')
    val_loader = dataloader(X_val,y_val)

    criterion = torch.nn.L1Loss()

    # Modified optimizer configuration
    optimizer = torch.optim.SGD(
        lmodel.parameters(), 
        lr=INIT_LR,
        momentum=0.9,  # Add momentum for smoother updates
        weight_decay=WEIGHT_DECAY  # Enable regularization
    )

    # Modified learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=epochs,  # Align with global training schedule
        eta_min=INIT_LR/100
    )

    # Add gradient clipping
    torch.nn.utils.clip_grad_norm_(lmodel.parameters(), max_norm=1.0)

    # Add alignment loss with global model
    alignment_weight = 0.3  # Controls global-local alignment strength
    
    global_model = copy.deepcopy(model).eval()  # Use current global model
    train_losses,val_losses = [],[]

    # Training loop modifications
    for epoch in range(epochs):
        lmodel.train()
        total_train_loss = 0
        total_alignment_loss = 0  # <-- ADDED

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            y_batch = y_batch.view(-1, 1)
            
            optimizer.zero_grad()
            predictions = lmodel(X_batch).view(-1, 1)
            
            # Original loss
            loss = criterion(predictions, y_batch)
            
            # Alignment loss with global model
            alignment_loss = 0  # <-- ADDED
            for (l_param, g_param) in zip(lmodel.parameters(), global_model.parameters()):
                alignment_loss += torch.norm(l_param - g_param.detach(), p=2)
            
            total_loss = loss + alignment_weight * alignment_loss  # <-- ADDED
            total_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(lmodel.parameters(), GRADIENT_CLIP)  # <-- ADDED
            
            optimizer.step()
            
            total_train_loss += loss.item()
            total_alignment_loss += alignment_loss.item()  # <-- ADDED

        # Changed scheduler step
        scheduler.step()  # <-- CHANGED
        train_loss = total_train_loss / len(train_loader)
        train_losses.append(train_loss)
        epoch_lgrads[f'{epoch+1}'] = compute_gradients(lmodel, criterion, X_train_raw, y_train, scaler, total_samples)

        # Validation
        lmodel.eval()
        total_val_loss = 0
        with torch.no_grad():
            for X_val, y_val in val_loader:
                X_val, y_val = X_val.to(DEVICE), y_val.to(DEVICE)
                # Ensure target shape
                y_val = y_val.view(-1, 1)

                val_predictions = lmodel(X_val)
                val_predictions = val_predictions.view(-1, 1)

                loss = criterion(val_predictions, y_val)
                total_val_loss += loss.item()

        val_loss = total_val_loss / len(val_loader)
        val_losses.append(val_loss)
   
    avg_train_loss = sum(train_losses)/len(train_losses)
    print(f'Train loss :{avg_train_loss} and Val loss :{val_loss} of silo name :{silo}')

    globalmodel = copy.deepcopy(lmodel)
    global_train_path = train_path.replace(f'{silo}/Train_{silo}.csv','eNki/Train_eNki.csv')
    X_global, y_global, silo = preprocess(global_train_path)

    X_train_raw,y_train,X_val_raw,y_val = split_data(X_global,y_global)
    
    global GLOBAL_SCALER
    GLOBAL_SCALER = None

    # Use a new scaler fit on global data (if distribution differs)
    if GLOBAL_SCALER is None:
        GLOBAL_SCALER = StandardScaler()

    X_train = GLOBAL_SCALER.fit_transform(X_train_raw)
    X_val = GLOBAL_SCALER.transform(X_val_raw)

    print(f"train dataset shape: {X_train.shape}")  # Should be (n_samples, 1073)
    print(f"validation dataset shape: {X_val.shape}")   # Should be (n_samples, 1073)
    
    # Convert to tensors
    train_loader = dataloader(X_train,y_train,'train')
    val_loader = dataloader(X_val,y_val)

    #optimizer = torch.optim.Adam(gmodel.parameters(), lr=INIT_LR, weight_decay=WEIGHT_DECAY)
    optimizer = optim.SGD(globalmodel.parameters(), lr=INIT_LR, momentum=0.9, weight_decay=WEIGHT_DECAY)
    criterion = torch.nn.L1Loss()

    # This scheduler reduces the learning rate by a factor of 0.5 if the average loss does not improve for 2 epochs.
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)

    gtrain_losses, gval_losses = [], []

    for epoch in range(epochs):
        globalmodel.train()
        total_train_loss = 0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
        
            # Ensure target shape is (batch_size, 1)
            y_batch = y_batch.view(-1, 1)
            optimizer.zero_grad()
            predictions = globalmodel(X_batch)

            # Ensure predictions match target shape
            predictions = predictions.view(-1, 1)

            loss = criterion(predictions, y_batch)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        train_loss = total_train_loss / len(train_loader)
        gtrain_losses.append(train_loss)

        # Validation
        globalmodel.eval()
        total_val_loss = 0
        with torch.no_grad():
            for X_val, y_val in val_loader:
                X_val, y_val = X_val.to(DEVICE), y_val.to(DEVICE)
                # Ensure target shape
                y_val = y_val.view(-1, 1)

                val_predictions = globalmodel(X_val)
                val_predictions = val_predictions.view(-1, 1)

                loss = criterion(val_predictions, y_val)
                total_val_loss += loss.item()

        val_loss = total_val_loss / len(val_loader)
        gval_losses.append(val_loss)

        # Step the scheduler with the validation loss.
        scheduler.step(val_loss)

        #print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        epoch_ggrads[f'{epoch+1}'] = compute_gradients(globalmodel, criterion, X_global, y_global, GLOBAL_SCALER, total_samples)
    
    avg_train_loss = sum(gtrain_losses)/len(gtrain_losses)
    avg_val_loss = sum(gval_losses)/len(gval_losses)

    print(f'Train loss :{avg_train_loss} and Val loss :{val_loss} of silo name :{silo}')
    aggregated_grads = harmonize_localglobal(lmodel,epoch_lgrads,epoch_ggrads)

    return aggregated_grads, train_losses, val_losses,scaler

def train_centralmodel(train_path, max_epochs):
    X, y, _ = preprocess(train_path)
    X_train_raw,y_train,X_val_raw,y_val = split_data(X,y)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_val = scaler.transform(X_val_raw)
    
    # Convert to tensors
    train_loader = dataloader(X_train,y_train,'train')
    val_loader = dataloader(X_val,y_val)
    
    model = AgePredictor(X_train.shape[1]).to(DEVICE)
    model.initialize_weights(model)
    criterion = torch.nn.L1Loss()
    optimizer = optim.SGD(model.parameters(), lr=1e-3, weight_decay=1e-5)
    train_losses, val_losses = [], []
    
    for epoch in range(max_epochs):
        model.train()
        total_train_loss = 0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
        
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
                X_val, y_val = X_val.to(DEVICE), y_val.to(DEVICE)
                # Ensure target shape
                y_val = y_val.view(-1, 1)

                val_predictions = model(X_val)
                val_predictions = val_predictions.view(-1, 1)

                loss = criterion(val_predictions, y_val)
                total_val_loss += loss.item()

        val_loss = total_val_loss / len(val_loader)
        val_losses.append(val_loss)

    return train_losses,val_losses, scaler,model