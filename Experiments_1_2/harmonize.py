import torch.nn as nn,numpy as np,pandas as pd,torch
from neuroCombat import neuroCombat

def harmonize_localsilos(model,gradients):

    silos = list(gradients.keys())
    aggregated_gradients ={}
    # Extracting layer shape directly
    for name, param in model.named_parameters():
        if name.startswith('fc') and name != 'fc3.bias': #print(f"Layer: {name}, Shape: {param.shape}")
            data_combined_list = []  # List to store combined gradients for each silo
            covars = {'batch': []}
            
            for silo in silos:
                data = gradients[silo]
                grad_list = []
                for epoch in list(data.keys()):
                    layer_grad = data[epoch][name].cpu().numpy().reshape(-1) # Flatten the gradient
                    grad_list.append(layer_grad)
                    # Uncomment for debugging:
                    # print(f'{silo} and Epoch :{epoch} and Layer :{name} and Shape: {layer_grad.shape}')

                stacked = np.stack(grad_list, axis=0) # Stack the gradients along a new axis

                #if name == 'fc3.bias':
                #    stacked += np.random.randn(*stacked.shape) * 1e-6
                
                combined_array = stacked.T # Transpose to get the desired shape: (num_elements, num_epochs)
                print(f'gradient map: {combined_array.shape}')
                
                # Append this combined array to our list
                data_combined_list.append(combined_array)
                
                # Determine the number of columns (samples) for this silo
                n_silo = combined_array.shape[1]
                # Append the current silo label for each sample in this silo
                covars['batch'].extend([silo] * n_silo)
            
            # Combine data from all silos along the column axis
            data_combined = np.hstack(data_combined_list)
            print("Combined data shape:", data_combined.shape)
            
            # Create the covariate DataFrame
            covars = pd.DataFrame(covars)
            
            # Harmonize the data using neuroCombat
            harmonized_output = neuroCombat(
                dat=data_combined,
                covars=covars,
                batch_col='batch',         # Column name for batch/scanner IDs
                categorical_cols=[],         # List any categorical variables if necessary
                eb=True,                     # Use Empirical Bayes
                parametric=True,             # Use parametric adjustment
                mean_only=False,             # Adjust both mean and variance
                ref_batch=None               # Harmonize to overall average (or set a specific reference batch)
            )
            #print("Harmonized output shape:", harmonized_output["data"].shape)
            data_harmonized = harmonized_output["data"]

            # Split harmonized data back into original scanners (if needed)
            harmonized_silo_1 = torch.from_numpy(np.mean(data_harmonized[:, :15],axis=1).reshape(param.shape).astype(np.float32)).to(param.device)
            harmonized_silo_2 = torch.from_numpy(np.mean(data_harmonized[:, 15:],axis=1).reshape(param.shape).astype(np.float32)).to(param.device)

            print(f'Layer :{name} silo 1:{harmonized_silo_1.shape} and silo 2:{harmonized_silo_2.shape}')
            aggregated_gradients[name] = harmonized_silo_1 + harmonized_silo_2

    return aggregated_gradients

def harmonize_localglobal(model,local_grads,global_grads):

    aggregated_gradients ={}
    # Extracting layer shape directly
    for name, param in model.named_parameters():
        if name.startswith('fc') and name != 'fc3.bias': #print(f"Layer: {name}, Shape: {param.shape}")
            data_combined_list = []  # List to store combined gradients for each silo
            covars = {'batch': []}
            
            lgrad_list,ggrad_list = [],[]
            for epoch in list(local_grads.keys()):
                layer_grad = local_grads[epoch][name].cpu().numpy().reshape(-1) # Flatten the gradient
                lgrad_list.append(layer_grad)
            
            stacked = np.stack(lgrad_list, axis=0) # Stack the gradients along a new axis

            combined_array = stacked.T # Transpose to get the desired shape: (num_elements, num_epochs)
            print(f'local gradient map: {combined_array.shape}')
            # Append this combined array to our list
            data_combined_list.append(combined_array)

            # Determine the number of columns (samples) for this silo
            n_silo = combined_array.shape[1]
            # Append the current silo label for each sample in this silo
            covars['batch'].extend(['local'] * n_silo)
            
            for epoch in list(global_grads.keys()):
                layer_grad = global_grads[epoch][name].cpu().numpy().reshape(-1) # Flatten the gradient
                ggrad_list.append(layer_grad)
                # Uncomment for debugging:
                # print(f'{silo} and Epoch :{epoch} and Layer :{name} and Shape: {layer_grad.shape}')

            stacked = np.stack(ggrad_list, axis=0) # Stack the gradients along a new axis

            combined_array = stacked.T # Transpose to get the desired shape: (num_elements, num_epochs)
            print(f'global gradient map: {combined_array.shape}')
            
            # Append this combined array to our list
            data_combined_list.append(combined_array)
            
            # Determine the number of columns (samples) for this silo
            n_silo = combined_array.shape[1]
            # Append the current silo label for each sample in this silo
            covars['batch'].extend(['global'] * n_silo)
        
            # Combine data from all silos along the column axis
            data_combined = np.hstack(data_combined_list)
            print("Combined data shape:", data_combined.shape)
            
            # Create the covariate DataFrame
            covars = pd.DataFrame(covars)
            
            # Harmonize the data using neuroCombat
            harmonized_output = neuroCombat(
                dat=data_combined,
                covars=covars,
                batch_col='batch',         # Column name for batch/scanner IDs
                categorical_cols=[],         # List any categorical variables if necessary
                eb=True,                     # Use Empirical Bayes
                parametric=True,             # Use parametric adjustment
                mean_only=False,             # Adjust both mean and variance
                ref_batch=None               # Harmonize to overall average (or set a specific reference batch)
            )
            #print("Harmonized output shape:", harmonized_output["data"].shape)
            data_harmonized = harmonized_output["data"]

            # Split harmonized data back into original scanners (if needed)
            harmonized_silo_1 = torch.from_numpy(np.mean(data_harmonized[:, :15],axis=1).reshape(param.shape).astype(np.float32)).to(param.device)
            harmonized_silo_2 = torch.from_numpy(np.mean(data_harmonized[:, 15:],axis=1).reshape(param.shape).astype(np.float32)).to(param.device)

            print(f'Layer :{name} local :{harmonized_silo_1.shape} and global :{harmonized_silo_2.shape}')
            aggregated_gradients[name] = harmonized_silo_1 + harmonized_silo_2
        #print('\n')

    return aggregated_gradients