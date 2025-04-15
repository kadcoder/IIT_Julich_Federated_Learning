#%%
import torch.nn as nn
import numpy as np
import pandas as pd
import torch
from typing import Dict, Any
from neuroCombat import neuroCombat
#%%
def harmonize_localsilos(
    model: nn.Module,
    gradients: Dict[str, Dict[int, Dict[str, torch.Tensor]]]
) -> Dict[str, torch.Tensor]:
    """
    Harmonizes gradients from multiple silos using neuroCombat to mitigate site/batch effects.

    Parameters:
    -----------
    model : torch.nn.Module
        The model containing the parameters (layers) whose gradients are to be harmonized.

    gradients : dict
        Dictionary of silo-specific gradients. Format:
        {
            'silo_name_1': {
                epoch1: {layer_name: tensor, ...},
                epoch2: {...},
                ...
            },
            'silo_name_2': {...},
            ...
        }

    Returns:
    --------
    aggregated_gradients : dict
        Dictionary of harmonized gradients aggregated across silos, keyed by layer names.
    """

    silos = list(gradients.keys())
    aggregated_gradients = {}

    for name, param in model.named_parameters():
        # Harmonize only fully connected layers (excluding last layer's bias)
        if name.startswith('fc') and name != 'fc3.bias':
            data_combined_list = []  # List to store flattened gradients per silo
            covars = {'batch': []}   # Covariate info for neuroCombat

            for silo in silos:
                data = gradients[silo]
                grad_list = []

                # Collect flattened gradients across epochs for this layer
                for epoch in list(data.keys()):
                    layer_grad = data[epoch][name].cpu().numpy().reshape(-1)
                    grad_list.append(layer_grad)

                # Stack epoch-wise gradients: shape (num_epochs, num_params)
                stacked = np.stack(grad_list, axis=0)

                # Transpose to shape (num_params, num_epochs)
                combined_array = stacked.T
                print(f'gradient map: {combined_array.shape}')

                # Store this silo's gradients and label all its samples
                data_combined_list.append(combined_array)
                covars['batch'].extend([silo] * combined_array.shape[1])

            # Concatenate gradients from all silos: shape (num_params, total_samples)
            data_combined = np.hstack(data_combined_list)
            print("Combined data shape:", data_combined.shape)

            # Create DataFrame for neuroCombat covariates
            covars = pd.DataFrame(covars)

            # Apply neuroCombat harmonization
            harmonized_output = neuroCombat(
                dat=data_combined,
                covars=covars,
                batch_col='batch',
                categorical_cols=[],
                eb=True,
                parametric=True,
                mean_only=False,
                ref_batch=None
            )
            
            data_harmonized = harmonized_output["data"]

            # Average harmonized gradients per silo (assuming two silos, each with 15 epochs)
            harmonized_silo_1 = torch.from_numpy(
                np.mean(data_harmonized[:, :15], axis=1).reshape(param.shape).astype(np.float32)
            ).to(param.device)

            harmonized_silo_2 = torch.from_numpy(
                np.mean(data_harmonized[:, 15:], axis=1).reshape(param.shape).astype(np.float32)
            ).to(param.device)

            print(f'Layer: {name} silo 1: {harmonized_silo_1.shape}, silo 2: {harmonized_silo_2.shape}')

            # Aggregate harmonized gradients by summing
            aggregated_gradients[name] = harmonized_silo_1 + harmonized_silo_2

    return aggregated_gradients

def harmonize_localglobal(
    model: nn.Module,
    local_grads: Dict[int, Dict[str, torch.Tensor]],
    global_grads: Dict[int, Dict[str, torch.Tensor]]
) -> Dict[str, torch.Tensor]:
    """
    Harmonizes local and global gradients for each layer using neuroCombat to mitigate
    domain-specific bias (e.g., site effects between local and global updates).

    This function targets specific layers (fully connected layers excluding 'fc3.bias') and:
    1. Collects gradients from both local and global updates across training epochs.
    2. Applies neuroCombat to harmonize the gradient distributions across 'local' and 'global' batches.
    3. Averages the harmonized gradients per domain and aggregates them by summation.
    
    Parameters:
    -----------
    model : nn.Module
        The PyTorch model containing named parameters (layers).

    local_grads : dict
        Dictionary of local gradients across epochs.
        Format: {
            epoch1: {layer_name: tensor, ...},
            epoch2: {...},
            ...
        }

    global_grads : dict
        Dictionary of global gradients across epochs (same format as local_grads).

    Returns:
    --------
    aggregated_gradients : dict
        Dictionary of harmonized and aggregated gradients, keyed by layer name.
        Format: {
            layer_name: aggregated_gradient_tensor,
            ...
        }
    """
    
    aggregated_gradients: Dict[str, torch.Tensor] = {}

    for name, param in model.named_parameters():
        # Harmonize only selected layers
        if name.startswith('fc') and name != 'fc3.bias':
            data_combined_list = []  # Store gradients for 'local' and 'global'
            covars = {'batch': []}

            # Process local gradients
            lgrad_list = []
            for epoch in local_grads:
                grad = local_grads[epoch][name].cpu().numpy().reshape(-1)
                lgrad_list.append(grad)

            stacked_local = np.stack(lgrad_list, axis=0).T  # Shape: (num_params, num_epochs)
            print(f'Local gradient map: {stacked_local.shape}')
            data_combined_list.append(stacked_local)
            covars['batch'].extend(['local'] * stacked_local.shape[1])

            # Process global gradients
            ggrad_list = []
            for epoch in global_grads:
                grad = global_grads[epoch][name].cpu().numpy().reshape(-1)
                ggrad_list.append(grad)

            stacked_global = np.stack(ggrad_list, axis=0).T  # Shape: (num_params, num_epochs)
            print(f'Global gradient map: {stacked_global.shape}')
            data_combined_list.append(stacked_global)
            covars['batch'].extend(['global'] * stacked_global.shape[1])

            # Combine local and global gradients
            data_combined = np.hstack(data_combined_list)
            print("Combined data shape:", data_combined.shape)

            covars_df = pd.DataFrame(covars)

            # Apply neuroCombat harmonization
            harmonized_output = neuroCombat(
                dat=data_combined,
                covars=covars_df,
                batch_col='batch',
                categorical_cols=[],
                eb=True,
                parametric=True,
                mean_only=False,
                ref_batch=None
            )

            data_harmonized = harmonized_output["data"]

            # Assuming equal number of epochs per group
            num_epochs = stacked_local.shape[1]
            harmonized_local = torch.from_numpy(
                np.mean(data_harmonized[:, :num_epochs], axis=1).reshape(param.shape).astype(np.float32)
            ).to(param.device)

            harmonized_global = torch.from_numpy(
                np.mean(data_harmonized[:, num_epochs:], axis=1).reshape(param.shape).astype(np.float32)
            ).to(param.device)

            print(f'Layer: {name} | Local: {harmonized_local.shape} | Global: {harmonized_global.shape}')

            # Aggregate harmonized gradients
            aggregated_gradients[name] = harmonized_local + harmonized_global

    return aggregated_gradients
