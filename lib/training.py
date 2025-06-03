#%%
import torch
import torch.nn as nn
from typing import Tuple, List, Dict
import numpy as np
import torch.optim as optim
import copy
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from lib.model import AgePredictor
from lib.data_utils import preprocess, dataloader, compute_gradients
from lib.harmonize import harmonize_localglobal
#%%

def gaussian_kernel(x: torch.Tensor,
                    y: torch.Tensor,
                    kernel_mul: float = 2.0,
                    kernel_num: int = 5,
                    fix_sigma: float = None) -> torch.Tensor:
    """Builds a multi‐bandwidth Gaussian kernel matrix between x and y."""
    n_x = x.size(0)
    total = torch.cat([x, y], dim=0)
    L2_dist = (
        total.pow(2).sum(1, keepdim=True)
        - 2 * total @ total.t()
        + total.pow(2).sum(1, keepdim=True).t()
    )
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.median(L2_dist.detach())
        bandwidth = bandwidth / (kernel_mul ** (kernel_num // 2))
    sigmas = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
    kernel_vals = [torch.exp(-L2_dist / sigma) for sigma in sigmas]
    return sum(kernel_vals)

def mmd2(x: torch.Tensor,
         y: torch.Tensor,
         kernel_mul: float = 2.0,
         kernel_num: int = 5,
         fix_sigma: float = None) -> torch.Tensor:
    """
    Computes squared‐MMD between x and y:
      MMD^2 = E[k(x,x)] + E[k(y,y)] - 2 E[k(x,y)]
    """
    batch_size = x.size(0)
    kernels = gaussian_kernel(x, y, kernel_mul, kernel_num, fix_sigma)
    XX = kernels[:batch_size, :batch_size]
    YY = kernels[batch_size:, batch_size:]
    XY = kernels[:batch_size, batch_size:]
    return XX.mean() + YY.mean() - 2 * XY.mean()

def gaussian_kl(mu_p, var_p, mu_t, var_t, eps=1e-8):
    """Compute KL divergence between two Gaussians KL(p||t)"""
    return 0.5 * (
        torch.log((var_t + eps) / (var_p + eps)) 
        + (var_p + (mu_p - mu_t)**2) / (var_t + eps)
        - 1
    )

def train_localmodel(
    X_silo: np.ndarray,
    y_silo: np.ndarray,
    model: nn.Module,
    total_samples: int,
    epochs: int,
    best_val_loss: float,
    DEVICE: str,
    mu_prox : float = 0.05,            # weight for proximal term
    lambda_kl: float = 0.3,            # weight for KL divergence
    lambda_mmd: float = 0.5,
    GRADIENT_CLIP: float = 1.0,
    INIT_LR: float = 1e-3,
    WEIGHT_DECAY: float = 1e-5,        # weight for MMD regularizer
    kernel_mul: float = 2.0,
    kernel_num: int = 5,
    fix_sigma: float = None
) -> Tuple[dict, List[float], List[float], float, nn.Module]:
    """
    Trains a local model on silo-specific data with alignment to a global model.

    Parameters
    ----------
    X_silo : np.ndarray
        Feature matrix of the silo's data.
    y_silo : np.ndarray
        Target values corresponding to `X_silo`.
    model : nn.Module
        The current global model to initialize and align with.
    total_samples : int
        Total number of samples (used for computing gradients).
    epochs : int
        Number of epochs to train.
    best_val_loss : float
        Best validation loss observed so far, used for tracking best model.

    Returns
    -------
    local_gradients : dict
        Computed gradients for the locally trained model.
    train_losses : List[float]
        Training loss per epoch.
    val_losses : List[float]
        Validation loss per epoch.
    best_val_loss : float
        Updated best validation loss.
    best_model : nn.Module
        Best-performing model during training (based on validation loss).
    """

    # Split data into train/val
    X_train_inner, X_val_raw, y_train, y_val = train_test_split(
        X_silo, y_silo, test_size=0.3, random_state=42
    )

    # Normalize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_inner)
    X_val = scaler.transform(X_val_raw)

    lmodel = copy.deepcopy(model)
    print(f"train dataset shape: {X_train.shape}")
    print(f"validation dataset shape: {X_val.shape}")

    # Convert to dataloaders
    train_loader = dataloader(X_train, y_train, 'train')
    val_loader = dataloader(X_val, y_val)

    criterion = nn.L1Loss()
    optimizer = torch.optim.SGD(
        lmodel.parameters(), lr=INIT_LR, momentum=0.9, weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=INIT_LR / 100
    )

    # Gradient clipping value
    #torch.nn.utils.clip_grad_norm_(lmodel.parameters(), max_norm=1.0)

    alignment_weight = 0.3  # Regularization weight for global-local alignment
    
    global_model = copy.deepcopy(model).eval()  # Static reference
    best_model = None
    train_losses, val_losses = [], []

    for epoch in range(epochs):
        lmodel.train()
        total_train_loss = 0.0
        total_alignment_loss = 0.0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            y_batch = y_batch.view(-1, 1)
            proximal_term = 0.0

            optimizer.zero_grad()
            predictions = lmodel(X_batch).view(-1, 1)

            # Primary loss
            loss = criterion(predictions, y_batch)

            # Calculate Gaussian KL divergence
            mu_p = torch.mean(predictions)
            var_p = torch.var(predictions, unbiased=False)
            mu_t = torch.mean(y_batch)
            var_t = torch.var(y_batch, unbiased=False)
            
            kl_loss = gaussian_kl(mu_p, var_p, mu_t, var_t)

            # MMD^2 between preds and true ages
            loss_mmd = mmd2(predictions, y_batch,
                            kernel_mul=kernel_mul,
                            kernel_num=kernel_num,
                            fix_sigma=fix_sigma)


            # Alignment loss with global model
            alignment_loss = sum(torch.norm(l_param - g_param.detach(), p=2)
                for l_param, g_param in zip(lmodel.parameters(), global_model.parameters())) #/ len(list(lmodel.parameters()))
            
            for l_param, g_param in zip(lmodel.parameters(), global_model.parameters()):
                proximal_term += torch.norm(l_param - g_param.detach(), p=2)**2

            total_loss = loss + (mu_prox/2.0) * proximal_term + lambda_mmd * loss_mmd + lambda_kl * kl_loss #alignment_weight * alignment_loss
            total_loss.backward()

            # Clip gradients
            torch.nn.utils.clip_grad_norm_(lmodel.parameters(), GRADIENT_CLIP)
            optimizer.step()

            total_train_loss += total_loss.item()
            #total_alignment_loss += alignment_loss_mmd.item()

        scheduler.step()
        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation phase
        lmodel.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for X_val_batch, y_val_batch in val_loader:
                X_val_batch, y_val_batch = X_val_batch.to(DEVICE), y_val_batch.to(DEVICE)
                y_val_batch = y_val_batch.view(-1, 1)
                val_predictions = lmodel(X_val_batch).view(-1, 1)
                loss = criterion(val_predictions, y_val_batch)
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model = copy.deepcopy(lmodel)

    # Fallback in case no improvement
    if best_model is None:
        best_model = copy.deepcopy(model)

    # Compute gradients
    local_gradients = compute_gradients(lmodel, criterion, X_train, y_train, total_samples)

    print(f'Train loss :{sum(train_losses)/len(train_losses):.4f} and Val loss :{sum(val_losses)/len(val_losses):.4f}')

    return local_gradients, train_losses, val_losses, best_val_loss, best_model

def train_globalmodel(
    gmodel: nn.Module,
    global_train_path: str,
    DEVICE: str,
    INIT_LR: float = 1e-3,
    WEIGHT_DECAY: float = 1e-5,
    lambda_kl: float = 0.3,            # weight for KL divergence
    lambda_mmd: float = 0.5,            # weight for MMD regularizer
    kernel_mul: float = 2.0,
    kernel_num: int = 5,
    fix_sigma: float = None,
    FINE_TUNE_EPOCHS: int = 10
) -> Tuple[nn.Module, float, List[float], List[float]]:
    """
    Trains (fine-tunes) the global model using data from the provided path.

    Parameters
    ----------
    gmodel : nn.Module
        The PyTorch model to be trained globally.
    global_train_path : str
        Path to the global training dataset.
    total_samples : int
        Total number of samples across all data sources (used for consistency).
    FINE_TUNE_EPOCHS : int, optional
        Number of epochs to fine-tune the model, by default 10.

    Returns
    -------
    gmodel : nn.Module
        The trained global model.
    avg_val_loss : float
        The average validation loss across all epochs.
    train_losses : List[float]
        List of training losses recorded per epoch.
    val_losses : List[float]
        List of validation losses recorded per epoch.
    """

    global_dict = {}
    X_global, y_global, silo = preprocess(global_train_path)

    # Split into training and validation sets
    X_train_inner, X_val_raw, y_train, y_val = train_test_split(
        X_global, y_global, test_size=0.3, random_state=42
    )

    global_scaler = StandardScaler()
    X_train = global_scaler.fit_transform(X_train_inner)
    X_val = global_scaler.transform(X_val_raw)

    print(f"train dataset shape: {X_train.shape}")  # Expecting (n_samples, 1073)
    print(f"validation dataset shape: {X_val.shape}")

    # Convert to tensor-based dataloaders
    train_loader = dataloader(X_train, y_train, 'train')
    val_loader = dataloader(X_val, y_val)

    optimizer = optim.SGD(gmodel.parameters(), lr=INIT_LR, momentum=0.9,
                          weight_decay=WEIGHT_DECAY)
    criterion = nn.L1Loss()

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2, verbose=True
    )
    global_dict['val_dataloader'] = val_loader
    global_dict['criterion'] = criterion

    train_losses, val_losses = [], []
    best_model = None
    best_val_loss = float('inf')

    for epoch in range(FINE_TUNE_EPOCHS):
        gmodel.train()
        total_train_loss = 0.0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            y_batch = y_batch.view(-1, 1)

            optimizer.zero_grad()
            predictions = gmodel(X_batch).view(-1, 1)

            loss = criterion(predictions, y_batch)

            # MMD^2 between preds and true ages
            loss_mmd = mmd2(predictions, y_batch,
                            kernel_mul=kernel_mul,
                            kernel_num=kernel_num,
                            fix_sigma=fix_sigma)

            # Calculate Gaussian KL divergence
            mu_p = torch.mean(predictions)
            var_p = torch.var(predictions, unbiased=False)
            mu_t = torch.mean(y_batch)
            var_t = torch.var(y_batch, unbiased=False)
            
            kl_loss = gaussian_kl(mu_p, var_p, mu_t, var_t)

            # Add MMD loss to the primary loss
            total_loss = loss + lambda_kl * kl_loss + lambda_mmd * loss_mmd
            total_loss.backward()
            optimizer.step()

            total_train_loss += total_loss.item()

        train_loss = total_train_loss / len(train_loader)
        train_losses.append(train_loss)

        # Validation loop
        gmodel.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for X_val, y_val in val_loader:
                X_val, y_val = X_val.to(DEVICE), y_val.to(DEVICE)
                y_val = y_val.view(-1, 1)

                val_predictions = gmodel(X_val).view(-1, 1)
                val_loss = criterion(val_predictions, y_val)

                total_val_loss += val_loss.item()

        val_loss_epoch = total_val_loss / len(val_loader)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = copy.deepcopy(gmodel)

        val_losses.append(val_loss_epoch)

        scheduler.step(val_loss_epoch)

        #print(f"Epoch {epoch+1}/{FINE_TUNE_EPOCHS} - "
        #      f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss_epoch:.4f}")
    
    avg_val_loss = sum(val_losses) / len(val_losses)
    print(f'Avg loss on global data of {silo}: {sum(train_losses)/len(train_losses):.4f}')
    global_dict['validation_loss'] = best_val_loss

    return best_model, avg_val_loss, train_losses, val_losses, global_dict

def train_localmodelgrads(
    X_silo: np.ndarray,
    y_silo: np.ndarray,
    model: nn.Module,
    total_samples: int,
    epochs: int,
    best_val_loss: float,
    DEVICE: str,
    mu_prox : float = 0.05,            # weight for proximal term
    lambda_kl: float = 0.3,            # weight for KL divergence
    lambda_mmd: float = 0.5,
    GRADIENT_CLIP: float = 1.0,
    INIT_LR: float = 1e-3,
    WEIGHT_DECAY: float = 1e-5,            # weight for MMD regularizer
    kernel_mul: float = 2.0,
    kernel_num: int = 5,
    fix_sigma: float = None
    ) -> Tuple[Dict[str, dict], List[float], List[float], float, nn.Module]:
    """
    Trains a local model on silo-specific data, records parameter gradients
    at each epoch, and incorporates alignment regularization with the global model.

    Parameters
    ----------
    X_silo : np.ndarray
        Feature matrix of the silo's dataset.
    y_silo : np.ndarray
        Corresponding target values.
    model : nn.Module
        The current global model to initialize and align with.
    total_samples : int
        Total number of samples (used to scale gradients).
    epochs : int
        Number of training epochs.
    best_val_loss : float
        Best validation loss observed so far.

    Returns
    -------
    epoch_lgrads : Dict[str, dict]
        Dictionary mapping epoch number to the local gradients computed at that epoch.
    train_losses : List[float]
        Training loss per epoch.
    val_losses : List[float]
        Validation loss per epoch.
    best_val_loss : float
        Updated best validation loss.
    best_model : nn.Module
        Model achieving the lowest validation loss during training.
    """

    lmodel = copy.deepcopy(model)

    # Split into training and validation sets
    X_train_inner, X_val_raw, y_train, y_val = train_test_split(
        X_silo, y_silo, test_size=0.3, random_state=42
    )

    epoch_lgrads: Dict[str, dict] = {}

    # Normalize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_inner)
    X_val = scaler.transform(X_val_raw)

    print(f"train dataset shape: {X_train.shape}")
    print(f"validation dataset shape: {X_val.shape}")

    # Convert to dataloaders
    train_loader = dataloader(X_train, y_train, 'train')
    val_loader = dataloader(X_val, y_val)

    criterion = nn.L1Loss()
    optimizer = torch.optim.SGD(lmodel.parameters(), lr=INIT_LR, momentum=0.9, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=INIT_LR / 100)

    # Gradient clipping value
    #torch.nn.utils.clip_grad_norm_(lmodel.parameters(), max_norm=1.0)

    alignment_weight = 0.5
    global_model = copy.deepcopy(model).eval()  # Freeze for alignment
    best_model = None
    train_losses: List[float] = []
    val_losses: List[float] = []

    for epoch in range(epochs):
        lmodel.train()
        total_train_loss = 0.0
        total_alignment_loss = 0.0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            y_batch = y_batch.view(-1, 1)
            proximal_term = 0.0

            optimizer.zero_grad()
            predictions = lmodel(X_batch).view(-1, 1)

            loss = criterion(predictions, y_batch)

            # Calculate Gaussian KL divergence
            mu_p = torch.mean(predictions)
            var_p = torch.var(predictions, unbiased=False)
            mu_t = torch.mean(y_batch)
            var_t = torch.var(y_batch, unbiased=False)
        
            kl_loss = gaussian_kl(mu_p, var_p, mu_t, var_t)

            # MMD^2 between preds and true ages
            loss_mmd = mmd2(predictions, y_batch,
                            kernel_mul=kernel_mul,
                            kernel_num=kernel_num,
                            fix_sigma=fix_sigma)

            
            alignment_loss = sum(
                torch.norm(l_param - g_param.detach(), p=2)
                for l_param, g_param in zip(lmodel.parameters(), global_model.parameters())) #/ len(list(lmodel.parameters()))
            #print(f"Predictions:{predictions} and alignment_loss: {alignment_loss}")
            for l_param, g_param in zip(lmodel.parameters(), global_model.parameters()):
                proximal_term += torch.norm(l_param - g_param.detach(), p=2)**2

            total_loss = loss + (mu_prox/2.0) * proximal_term + lambda_kl * kl_loss + lambda_mmd * loss_mmd #+alignment_weight * alignment_loss
            total_loss.backward()
            
            torch.nn.utils.clip_grad_norm_(lmodel.parameters(), GRADIENT_CLIP)
            
            optimizer.step()

            total_train_loss += total_loss.item()
            #total_alignment_loss += alignment_loss.item()

        scheduler.step()
        train_loss = total_train_loss / len(train_loader)
        train_losses.append(train_loss)

        # Store local gradients at the current epoch
        epoch_lgrads[str(epoch + 1)] = compute_gradients(
            lmodel, criterion, X_train, y_train, total_samples
        )

        # Validation step
        lmodel.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for X_val_batch, y_val_batch in val_loader:
                X_val_batch, y_val_batch = X_val_batch.to(DEVICE), y_val_batch.to(DEVICE)
                y_val_batch = y_val_batch.view(-1, 1)
                val_predictions = lmodel(X_val_batch).view(-1, 1)
                loss = criterion(val_predictions, y_val_batch)
                total_val_loss += loss.item()

        val_loss = total_val_loss / len(val_loader)
        val_losses.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = copy.deepcopy(lmodel)

    avg_train_loss = sum(train_losses) / len(train_losses)
    avg_val_loss = sum(val_losses) / len(val_losses)

    print(f'Train loss :{avg_train_loss:.4f} and Val loss :{avg_val_loss:.4f}')

    if best_model is None:
        best_model = copy.deepcopy(model)  # fallback to last model

    return epoch_lgrads, train_losses, val_losses, best_val_loss, best_model

def train_localglobalgrads(
    X_silo: np.ndarray, 
    y_silo: np.ndarray, 
    X_global: np.ndarray, 
    y_global: np.ndarray, 
    model: torch.nn.Module, 
    total_samples: int, 
    epochs: int,
    best_val_loss: float,
    DEVICE: str,
    mu_prox : float = 0.05,            # weight for proximal term
    lambda_kl: float = 0.3,            # weight for KL divergence
    lambda_mmd: float = 0.5,
    lambda_kl_g: float = 0.3,            # weight for KL divergence
    lambda_mmd_g: float = 0.5,
    GRADIENT_CLIP: float = 1.0,
    INIT_LR: float = 1e-3,
    WEIGHT_DECAY: float = 1e-5,            # weight for MMD regularizer
    kernel_mul: float = 2.0,
    kernel_num: int = 5,
    fix_sigma: float = None) -> Tuple[Dict[str, torch.Tensor], List[float], List[float], float, torch.nn.Module]:
    """
    Trains a local model with alignment to the global model, then updates the global model
    using a subset of centralized data. Returns harmonized gradients and training logs.

    Args:
        X_silo (Any): Local training features for the silo.
        y_silo (Any): Local training labels for the silo.
        X_global (Any): Global (centralized) dataset features.
        y_global (Any): Global dataset labels.
        model (torch.nn.Module): Initial global model.
        total_samples (int): Total number of samples across all silos.
        epochs (int): Number of training epochs.
        best_val_loss (float): Best observed validation loss (for model selection).

    Returns:
        Tuple: (aggregated_grads, train_losses, val_losses, best_val_loss, best_model)
            - aggregated_grads (dict): Harmonized gradients from local and global training.
            - train_losses (list): Per-epoch training losses on local data.
            - val_losses (list): Per-epoch validation losses on local data.
            - best_val_loss (float): Updated best validation loss.
            - best_model (torch.nn.Module): Best performing model.
    """
    lmodel = copy.deepcopy(model)

    X_train_inner, X_val_raw, y_train, y_val = train_test_split(
        X_silo, y_silo, test_size=0.3, random_state=42
    )

    epoch_lgrads, epoch_ggrads = {}, {}

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_inner)
    X_val = scaler.transform(X_val_raw)

    print(f"train dataset shape: {X_train.shape}")
    print(f"validation dataset shape: {X_val.shape}")

    train_loader = dataloader(X_train, y_train, 'train')
    val_loader = dataloader(X_val, y_val)

    criterion = torch.nn.L1Loss()
    optimizer = torch.optim.SGD(
        lmodel.parameters(), lr=INIT_LR, momentum=0.9, weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=INIT_LR / 100
    )

    alignment_weight = 0.5
    global_model = copy.deepcopy(model).eval()
    best_model = None
    train_losses, val_losses = [], []


    for epoch in range(epochs):
        lmodel.train()
        total_train_loss = 0
        total_alignment_loss = 0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            y_batch = y_batch.view(-1, 1)
            proximal_term = 0.0

            optimizer.zero_grad()
            predictions = lmodel(X_batch).view(-1, 1)
            loss = criterion(predictions, y_batch)

            # Calculate Gaussian KL divergence
            mu_p = torch.mean(predictions)
            var_p = torch.var(predictions, unbiased=False)
            mu_t = torch.mean(y_batch)
            var_t = torch.var(y_batch, unbiased=False)
        
            kl_loss = gaussian_kl(mu_p, var_p, mu_t, var_t)
            
            # MMD^2 between preds and true ages
            loss_mmd = mmd2(predictions, y_batch,
                            kernel_mul=kernel_mul,
                            kernel_num=kernel_num,
                            fix_sigma=fix_sigma)
           
            alignment_loss = sum(
                torch.norm(l_param - g_param.detach(), p=2)
                for l_param, g_param in zip(lmodel.parameters(), global_model.parameters())) #/ len(list(lmodel.parameters()))
            #print(f"Predictions:{predictions} and alignment_loss: {alignment_loss}")

            for l_param, g_param in zip(lmodel.parameters(), global_model.parameters()):
                proximal_term += torch.norm(l_param - g_param.detach(), p=2)**2
            
            total_loss = loss + (mu_prox/2.0) * proximal_term + lambda_kl * kl_loss + lambda_mmd * loss_mmd #alignment_weight * alignment_loss +
            total_loss.backward()
            
            torch.nn.utils.clip_grad_norm_(lmodel.parameters(), GRADIENT_CLIP)
            
            optimizer.step()

            total_train_loss += total_loss.item()
            #total_alignment_loss += alignment_loss_mmd.item()

        scheduler.step()
        train_loss = total_train_loss / len(train_loader)
        train_losses.append(train_loss)
        epoch_lgrads[f'{epoch+1}'] = compute_gradients(lmodel, criterion, X_train, y_train, total_samples)

        lmodel.eval()
        total_val_loss = 0
        with torch.no_grad():
            for X_val_batch, y_val_batch in val_loader:
                X_val_batch, y_val_batch = X_val_batch.to(DEVICE), y_val_batch.to(DEVICE)
                y_val_batch = y_val_batch.view(-1, 1)
                predictions = lmodel(X_val_batch).view(-1, 1)
                loss = criterion(predictions, y_val_batch)
                total_val_loss += loss.item()

        val_loss = total_val_loss / len(val_loader)
        val_losses.append(val_loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = copy.deepcopy(lmodel)

    if best_model is None:
        best_model = copy.deepcopy(model)

    avg_train_loss = sum(train_losses) / len(train_losses)
    avg_val_loss = sum(val_losses) / len(val_losses)
    print(f'Train loss :{avg_train_loss} and Val loss :{avg_val_loss}')

    globalmodel = copy.deepcopy(lmodel)

    X_train_inner, X_val_raw, y_train, y_val = train_test_split(
        X_global, y_global, test_size=0.3, random_state=42
    )

    global_scaler = StandardScaler()
    X_train = global_scaler.fit_transform(X_train_inner)
    X_val = global_scaler.transform(X_val_raw)

    print(f"train dataset shape: {X_train.shape}")
    print(f"validation dataset shape: {X_val.shape}")

    train_loader = dataloader(X_train, y_train, 'train')
    val_loader = dataloader(X_val, y_val)

    optimizer = optim.SGD(globalmodel.parameters(), lr=INIT_LR, momentum=0.9, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)

    gtrain_losses, gval_losses = [], []
    
    for epoch in range(epochs):
        globalmodel.train()
        total_train_loss = 0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            y_batch = y_batch.view(-1, 1)
            optimizer.zero_grad()
            predictions = globalmodel(X_batch).view(-1, 1)
            loss = criterion(predictions, y_batch)

            # Calculate Gaussian KL divergence
            mu_p = torch.mean(predictions)
            var_p = torch.var(predictions, unbiased=False)
            mu_t = torch.mean(y_batch)
            var_t = torch.var(y_batch, unbiased=False)
        
            kl_loss = gaussian_kl(mu_p, var_p, mu_t, var_t)

            # MMD^2 between preds and true ages
            loss_mmd = mmd2(predictions, y_batch,
                            kernel_mul=kernel_mul,
                            kernel_num=kernel_num,
                            fix_sigma=fix_sigma)
            
            # Add MMD loss to the primary loss
            total_loss = loss + lambda_kl_g * kl_loss + lambda_mmd_g * loss_mmd
            total_loss.backward()

            torch.nn.utils.clip_grad_norm_(globalmodel.parameters(), GRADIENT_CLIP)
            
            optimizer.step()
            total_train_loss += loss.item()

        train_loss = total_train_loss / len(train_loader)
        gtrain_losses.append(train_loss)
        epoch_ggrads[f'{epoch+1}'] = compute_gradients(globalmodel, criterion, X_train, y_train, total_samples)

        globalmodel.eval()
        total_val_loss = 0
        with torch.no_grad():
            for X_val_batch, y_val_batch in val_loader:
                X_val_batch, y_val_batch = X_val_batch.to(DEVICE), y_val_batch.to(DEVICE)
                y_val_batch = y_val_batch.view(-1, 1)
                val_predictions = globalmodel(X_val_batch).view(-1, 1)
                loss = criterion(val_predictions, y_val_batch)
                total_val_loss += loss.item()

        val_loss = total_val_loss / len(val_loader)
        #if val_loss < best_val_loss:
        #    best_val_loss = val_loss
        #    best_model = copy.deepcopy(globalmodel)

        gval_losses.append(val_loss)
        scheduler.step(val_loss)
    
    if best_model is None:
        best_model = copy.deepcopy(model)

    avg_train_loss = sum(gtrain_losses) / len(gtrain_losses)
    avg_val_loss = sum(gval_losses) / len(gval_losses)
    print(f'Train loss :{avg_train_loss} and Val loss :{avg_val_loss}')
    aggregated_grads = harmonize_localglobal(lmodel, epoch_lgrads, epoch_ggrads)

    return aggregated_grads, train_losses, val_losses, best_val_loss, best_model

def train_globalgrads(
    X_silo: np.ndarray, 
    y_silo: np.ndarray, 
    X_global: np.ndarray, 
    y_global: np.ndarray, 
    model: torch.nn.Module, 
    total_samples: int, 
    epochs: int,
    best_val_loss: float,
    DEVICE: str,
    mu_prox: float = 0.05,
    lambda_kl: float = 0.3,            # weight for KL divergence
    lambda_mmd: float = 0.5,
    lambda_kl_g:float =0.3,
    lambda_mmd_g=0.5,  
    GRADIENT_CLIP: float = 1.0,
    INIT_LR: float = 1e-3,
    WEIGHT_DECAY: float = 1e-5,# weight for MMD regularizer
    kernel_mul: float = 2.0,
    kernel_num: int = 5,
    fix_sigma: float = None) -> Tuple[Dict[str, torch.Tensor], List[float], List[float], float, torch.nn.Module]:
    """
    Trains a local model with alignment to the global model, then updates the global model
    using a subset of centralized data. Returns harmonized gradients and training logs.

    Args:
        X_silo (Any): Local training features for the silo.
        y_silo (Any): Local training labels for the silo.
        X_global (Any): Global (centralized) dataset features.
        y_global (Any): Global dataset labels.
        model (torch.nn.Module): Initial global model.
        total_samples (int): Total number of samples across all silos.
        epochs (int): Number of training epochs.
        best_val_loss (float): Best observed validation loss (for model selection).

    Returns:
        Tuple: (aggregated_grads, train_losses, val_losses, best_val_loss, best_model)
            - aggregated_grads (dict): Harmonized gradients from local and global training.
            - train_losses (list): Per-epoch training losses on local data.
            - val_losses (list): Per-epoch validation losses on local data.
            - best_val_loss (float): Updated best validation loss.
            - best_model (torch.nn.Module): Best performing model.
    """
    lmodel = copy.deepcopy(model)

    X_train_inner, X_val_raw, y_train, y_val = train_test_split(
        X_silo, y_silo, test_size=0.3, random_state=42
    )

    epoch_lgrads, epoch_ggrads = {}, {}

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_inner)
    X_val = scaler.transform(X_val_raw)

    print(f"train dataset shape: {X_train.shape}")
    print(f"validation dataset shape: {X_val.shape}")

    train_loader = dataloader(X_train, y_train, 'train')
    val_loader = dataloader(X_val, y_val)

    criterion = torch.nn.L1Loss()
    optimizer = torch.optim.SGD(lmodel.parameters(), lr=INIT_LR, momentum=0.9, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=INIT_LR / 100)

    alignment_weight = 0.3
    global_model = copy.deepcopy(model).eval()
    best_model = None
    train_losses, val_losses = [], []

    for epoch in range(epochs):
        lmodel.train()
        total_train_loss = 0
        total_alignment_loss = 0

        for X_batch, y_batch in train_loader:
            proximal_term = 0.0
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            y_batch = y_batch.view(-1, 1)

            optimizer.zero_grad()
            predictions = lmodel(X_batch).view(-1, 1)
            loss = criterion(predictions, y_batch)

            # Calculate Gaussian KL divergence
            mu_p = torch.mean(predictions)
            var_p = torch.var(predictions, unbiased=False)
            mu_t = torch.mean(y_batch)
            var_t = torch.var(y_batch, unbiased=False)
        
            kl_loss = gaussian_kl(mu_p, var_p, mu_t, var_t)

            # MMD^2 between preds and true ages
            loss_mmd = mmd2(predictions, y_batch,
                            kernel_mul=kernel_mul,
                            kernel_num=kernel_num,
                            fix_sigma=fix_sigma)
            
            for l_param, g_param in zip(lmodel.parameters(), global_model.parameters()):
                proximal_term += torch.norm(l_param - g_param.detach(), p=2)**2

            # FedLap-style layer-wise alignment
            #layer_norms = [torch.norm(lp - gp.detach(), p=2) for lp, gp in zip(lmodel.parameters(), global_model.parameters())]
            #norms_sum = sum([norm.item() for norm in layer_norms]) + 1e-8
            #alignment_loss = sum((norm.item() / norms_sum) * norm for norm in layer_norms)

            alignment_loss = sum(
                torch.norm(l_param - g_param.detach(), p=2)
                for l_param, g_param in zip(lmodel.parameters(), global_model.parameters())) #/ len(list(lmodel.parameters()))

            total_loss = loss + (mu_prox/2.0) * proximal_term + lambda_kl * kl_loss + lambda_mmd * loss_mmd #+ alignment_weight * alignment_loss
            total_loss.backward()

            torch.nn.utils.clip_grad_norm_(lmodel.parameters(), GRADIENT_CLIP)
            
            optimizer.step()

            total_train_loss += loss.item()
            #total_alignment_loss += alignment_loss_mmd.item()

        scheduler.step()
        train_loss = total_train_loss / len(train_loader)
        train_losses.append(train_loss)
        #epoch_lgrads[f'{epoch+1}'] = compute_gradients(lmodel, criterion, X_train, y_train, total_samples)

        lmodel.eval()
        total_val_loss = 0
        with torch.no_grad():
            for X_val_batch, y_val_batch in val_loader:
                X_val_batch, y_val_batch = X_val_batch.to(DEVICE), y_val_batch.to(DEVICE)
                y_val_batch = y_val_batch.view(-1, 1)
                predictions = lmodel(X_val_batch).view(-1, 1)
                loss = criterion(predictions, y_val_batch)
                total_val_loss += loss.item()

        val_loss = total_val_loss / len(val_loader)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = copy.deepcopy(lmodel)
        val_losses.append(val_loss)
    
    if best_model is None:
        best_model = copy.deepcopy(model)

    avg_train_loss = sum(train_losses) / len(train_losses)
    avg_val_loss = sum(val_losses) / len(val_losses)
    print(f'Train loss :{avg_train_loss} and Val loss :{avg_val_loss}')

    globalmodel = copy.deepcopy(lmodel) #if best_model is None else copy.deepcopy(best_model)

    X_train_inner, X_val_raw, y_train, y_val = train_test_split(
        X_global, y_global, test_size=0.3, random_state=42
    )

    global_scaler = StandardScaler()
    X_train = global_scaler.fit_transform(X_train_inner)
    X_val = global_scaler.transform(X_val_raw)

    print(f"train dataset shape: {X_train.shape}")
    print(f"validation dataset shape: {X_val.shape}")

    train_loader = dataloader(X_train, y_train, 'train')
    val_loader = dataloader(X_val, y_val)

    optimizer = optim.SGD(globalmodel.parameters(), lr=INIT_LR, momentum=0.9, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)

    gtrain_losses, gval_losses = [], []
    #best_model = None
    #best_val_loss = float('inf')
    
    for epoch in range(epochs//2):
        globalmodel.train()
        total_train_loss = 0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            y_batch = y_batch.view(-1, 1)
            optimizer.zero_grad()
            predictions = globalmodel(X_batch).view(-1, 1)
            loss = criterion(predictions, y_batch)

            # Calculate Gaussian KL divergence
            mu_p = torch.mean(predictions)
            var_p = torch.var(predictions, unbiased=False)
            mu_t = torch.mean(y_batch)
            var_t = torch.var(y_batch, unbiased=False)
        
            kl_loss = gaussian_kl(mu_p, var_p, mu_t, var_t)

            # MMD^2 between preds and true ages
            loss_mmd = mmd2(predictions, y_batch,
                            kernel_mul=kernel_mul,
                            kernel_num=kernel_num,
                            fix_sigma=fix_sigma)
            # Add MMD loss to the primary loss
            total_loss = loss + lambda_kl_g * kl_loss + lambda_mmd_g * loss_mmd
            total_loss.backward()
            
            torch.nn.utils.clip_grad_norm_(globalmodel.parameters(), GRADIENT_CLIP)
           
            optimizer.step()
            total_train_loss += total_loss.item()

        train_loss = total_train_loss / len(train_loader)
        gtrain_losses.append(train_loss)
        epoch_ggrads[f'{epoch+1}'] = compute_gradients(globalmodel, criterion, X_train, y_train, total_samples)

        globalmodel.eval()
        total_val_loss = 0
        with torch.no_grad():
            for X_val_batch, y_val_batch in val_loader:
                X_val_batch, y_val_batch = X_val_batch.to(DEVICE), y_val_batch.to(DEVICE)
                y_val_batch = y_val_batch.view(-1, 1)
                val_predictions = globalmodel(X_val_batch).view(-1, 1)
                loss = criterion(val_predictions, y_val_batch)
                total_val_loss += loss.item()

        val_loss = total_val_loss / len(val_loader)
        #if val_loss < best_val_loss:
        #    best_val_loss = val_loss
        #    best_model = copy.deepcopy(global_model)
        gval_losses.append(val_loss)
        scheduler.step(val_loss)
    
    #if best_model is None:
    #    best_model = copy.deepcopy(model)
    avg_train_loss = sum(gtrain_losses) / len(gtrain_losses)
    avg_val_loss = sum(gval_losses) / len(gval_losses)
    print(f'Train loss :{avg_train_loss} and Val loss :{avg_val_loss}')

    return epoch_ggrads, train_losses, val_losses, best_val_loss, best_model

def train_centralmodel(train_path, 
    max_epochs: int, 
    DEVICE:str,
    lambda_kl: float = 0.3,            # weight for KL divergence
    lambda_mmd: float = 0.5,            # weight for MMD regularizer
    kernel_mul: float = 2.0,
    kernel_num: int = 5,
    fix_sigma: float = None
    ) -> Tuple[List[float], List[float], StandardScaler, torch.nn.Module]:
    """
    Trains a centralized model on all available data and returns loss history, scaler, and best model.

    Args:
        train_path (str): Path to the training dataset.
        max_epochs (int): Number of training epochs.

    Returns:
        Tuple: (train_losses, val_losses, scaler, best_model)
            - train_losses (List[float]): Per-epoch training losses.
            - val_losses (List[float]): Per-epoch validation losses.
            - scaler (StandardScaler): Fitted scaler for feature normalization.
            - best_model (torch.nn.Module): Best performing model on validation data.
    """
    X, y, _ = preprocess(train_path)

    X_train_inner, X_val_raw, y_train, y_val = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_inner)
    X_val = scaler.transform(X_val_raw)

    train_loader = dataloader(X_train, y_train, 'train')
    val_loader = dataloader(X_val, y_val)

    model = AgePredictor(X_train.shape[1]).to(DEVICE)
    model.initialize_weights(model)

    criterion = torch.nn.L1Loss()
    optimizer = optim.SGD(model.parameters(), lr=1e-3, weight_decay=1e-5)

    train_losses, val_losses = [], []
    best_val_loss = float('inf')
    best_model = None

    for epoch in range(max_epochs):
        model.train()
        total_train_loss = 0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            y_batch = y_batch.view(-1, 1)

            optimizer.zero_grad()
            predictions = model(X_batch).view(-1, 1)
            loss = criterion(predictions, y_batch)

            # Calculate Gaussian KL divergence
            mu_p = torch.mean(predictions)
            var_p = torch.var(predictions, unbiased=False)
            mu_t = torch.mean(y_batch)
            var_t = torch.var(y_batch, unbiased=False)
            
            kl_loss = gaussian_kl(mu_p, var_p, mu_t, var_t)

            # MMD^2 between preds and true ages
            loss_mmd_gaus = mmd2(predictions, y_batch,kernel_mul=kernel_mul,kernel_num=kernel_num,fix_sigma=fix_sigma)

            # Calculate MMD loss
            #loss_mmd_imq = mmd2_with_kernel(predictions, y_batch, imq_kernel, C=2.0)
            #loss_mmd_lap = mmd2_with_kernel(predictions, y_batch, laplacian_kernel, sigma=0.5)
            
            # Add MMD loss to the primary loss
            total_loss = loss + lambda_kl * kl_loss + lambda_mmd * loss_mmd_gaus
            total_loss.backward()
            optimizer.step()
            total_train_loss += total_loss.item()

        train_losses.append(total_train_loss / len(train_loader))

        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for X_val_batch, y_val_batch in val_loader:
                X_val_batch, y_val_batch = X_val_batch.to(DEVICE), y_val_batch.to(DEVICE)
                y_val_batch = y_val_batch.view(-1, 1)
                predictions = model(X_val_batch).view(-1, 1)
                loss = criterion(predictions, y_val_batch)
                total_val_loss += loss.item()

        val_loss = total_val_loss / len(val_loader)
        val_losses.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = copy.deepcopy(model)

    return train_losses, val_losses, scaler, best_model