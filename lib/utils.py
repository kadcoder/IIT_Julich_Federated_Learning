#%%
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Union
#%%

def set_parameters(SEED: int = 42) -> None:
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def select_device(preferred_id=None):
    """
    Returns a torch.device on the first valid CUDA index (or the user-specified one),
    or CPU if CUDA isn’t available or the index is out of range.
    """
    if not torch.cuda.is_available():
        print("CUDA not available → using CPU")
        return torch.device("cpu")

    n_gpus = torch.cuda.device_count()
    print(f"Found {n_gpus} CUDA device(s):")
    for i in range(n_gpus):
        print(f"  [{i}]: {torch.cuda.get_device_name(i)}")

    # If the user passed a preferred_id and it’s in range, use it.
    if preferred_id is not None and 0 <= preferred_id < n_gpus:
        print(f"Using user-specified GPU: {preferred_id}")
        return torch.device(f"cuda:{preferred_id}")

    # Otherwise pick GPU 0 by default
    print("Falling back to GPU 0")
    return torch.device("cuda:0")

def ensure_dir(file_path: str) -> None:
    """
    Ensures the directory for the given file path exists.

    Args:
        file_path (str): The file path whose directory should be created.
    """
    dir_path = os.path.dirname(file_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)

def plot_loss_curves(
    loss_history: Dict[str, Dict[str, Union[List[float], np.ndarray]]],
    project_dir: str,
    experiment_subdir: str
) -> None:
    
    """
    Plots and saves the training and validation loss curves for multiple datasets.

    Parameters:
    -----------
    loss_history : dict
        Dictionary with keys like 'CamCAN' and 'SALD', each containing:
        {
            'train': list or array of training losses,
            'val': list or array of validation losses
        }

    project_dir : str
        Path to the root project directory.

    experiment_subdir : str
        Subdirectory under project_dir where the plot should be saved.
    """
    plot_path = os.path.join(project_dir, experiment_subdir)
    ensure_dir(plot_path)

    # Extract and flatten loss histories
    camcan_train = np.ravel(loss_history['CamCAN']['train'])
    camcan_val   = np.ravel(loss_history['CamCAN']['val'])
    sald_train   = np.ravel(loss_history['SALD']['train'])
    sald_val     = np.ravel(loss_history['SALD']['val'])

    # Epoch indices
    epochs_camcan = np.arange(1, len(camcan_train) + 1)
    epochs_sald   = np.arange(1, len(sald_train) + 1)

    # Plotting setup
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))

    # CamCAN Loss Plot
    axes[0].plot(epochs_camcan, camcan_train, label='Train Loss', color='blue', marker='o', markersize=2)
    axes[0].plot(epochs_camcan, camcan_val, label='Validation Loss', color='red', marker='o', markersize=2)
    axes[0].set_title("CamCAN Losses")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True)
    axes[0].set_xticks(np.arange(1, len(camcan_train) + 1, 200))

    # SALD Loss Plot
    axes[1].plot(epochs_sald, sald_train, label='Train Loss', color='blue', marker='o', markersize=2)
    axes[1].plot(epochs_sald, sald_val, label='Validation Loss', color='red', marker='o', markersize=2)
    axes[1].set_title("SALD Losses")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(True)
    axes[1].set_xticks(np.arange(1, len(sald_train) + 1, 200))

    # Save and show
    plt.tight_layout()
    save_path = os.path.join(plot_path, 'losses.png')
    plt.savefig(save_path)
    plt.show()
    plt.close(fig)

    print(f"[INFO] Loss plot saved at: {save_path}")