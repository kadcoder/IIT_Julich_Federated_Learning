import pandas as pd
import numpy as np
from neuroCombat import neuroCombat


def harmonize_localsilo(layers, silo1_grads, silo2_grads):

    harmonized_data = {}
    BN_layers = [layer for layer in layers if not layer.startswith('bn') and not layer.endswith('bias')]


    for layer in BN_layers:
        print(f"Harmonizing data for layer: {layer}")

        # Validate layer presence
        if layer not in silo1_grads or layer not in silo2_grads:
            raise KeyError(f"Layer {layer} missing in local or global gradients.")
        
        # Extract features
        local_features = np.array(silo1_grads[layer]['data'])
        global_features = np.array(silo2_grads[layer]['data'])

        print(f"Before reshape Local features {layer}: {local_features.shape}, Global features {layer}: {global_features.shape}")

        # Ensure 2D shape
        local_features = local_features.reshape(-1, 1) if local_features.ndim == 1 else local_features
        global_features = global_features.reshape(-1, 1) if global_features.ndim == 1 else global_features
        
        
        #if layer == 'fc3.weight':
            #local_features = local_features.T
            #global_features = global_features.T

        if(np.isnan(local_features).any()): print(f"Warning: NaN values found in local features : {layer}")
        if(np.isnan(global_features).any()): print(f"Warning: NaN values found in global features: {layer}")

        # For local features
        std_local = np.var(local_features, axis=0)
        if (std_local < 1e-6).all():
            noise_scale = 1e-3  # Significantly larger noise
            local_features += np.random.normal(0, noise_scale, local_features.shape)
            print(f"Added noise (scale={noise_scale}) to local features: {layer}")

        # For global features
        std_global = np.var(global_features, axis=0)
        if (std_global < 1e-6).all():
            noise_scale = 1e-3  # Same adjustment
            global_features += np.random.normal(0, noise_scale, global_features.shape)
            print(f"Added noise (scale={noise_scale}) to global features: {layer}")

        print(f"After reshape Local features {layer}: {local_features.shape}, Global features {layer}: {global_features.shape}")

        std_local = np.var(local_features,axis=0)
        if (std_local < 1e-6).all(): print(f"Warning: Very low variance in local features: {layer}")

        std_local = np.var(global_features,axis=0)
        if (std_local < 1e-6).all(): print(f"Warning: Very low variance in global features: {layer}")
        
        # Concatenate data
        try:
            combined_data = np.vstack([local_features, global_features])
            #std_comb = np.std(combined_data,axis=1)
            #if std_comb < 1e-6: print(f"Warning: Very low variance in local features: {layer}")

        except ValueError as e:
            print(f"Dimension mismatch in layer {layer}: {e}")
            continue

        # Prepare DataFrame
        feature_columns = [f"feature_{i}" for i in range(combined_data.shape[1])]
        data_df = pd.DataFrame(combined_data, columns=feature_columns)
        #print(f"dataframe:{data_df}")
        batch_labels = ['local'] * local_features.shape[0] + ['global'] * global_features.shape[0]
        covars = pd.DataFrame({'batch': batch_labels})
        
        try:
            # Apply NeuroCombat
            harmonized_results = neuroCombat(
                dat=data_df.T.values,  # (features, samples)
                covars=covars,
                batch_col='batch',
                categorical_cols=[]
            )
            
            if 'data' not in harmonized_results:
                print(f"No harmonized data for layer {layer}.")
                continue

            harmonized_values = harmonized_results['data'].T  # Convert back to (samples, features)
            harmonized_df = pd.DataFrame(harmonized_values, columns=feature_columns)
            harmonized_df['batch'] = batch_labels

            # Extract harmonized local features
            harmonized_localfeatures = harmonized_df[harmonized_df["batch"] == "local"].drop(columns=["batch"]).values

            # Transpose for specific layers
            #if layer == 'fc3.weight':
            #    harmonized_data[layer] = harmonized_localfeatures.T
            #else:
            harmonized_data[layer] = harmonized_localfeatures

            #if layer.endswith('.bias'):
            #    harmonized_data[layer] = harmonized_localfeatures.squeeze()
            
            #if(layer== 'fc3.bias'):
            #    print(f"bias:{harmonized_data[layer]}")


            print(f"Harmonized local features shape: {harmonized_data[layer].shape}")

            print(f"Successfully harmonized layer: {layer}\n")

            if(np.isnan(harmonized_data[layer]).any()): print(f"Warning: NaN values found in local features : {layer}")

            std_global = np.std(harmonized_data[layer])
            if std_global < 1e-6:
                print(f"Warning: Very low variance in global features: {layer}")
        
        except Exception as e:
            print(f"Error harmonizing layer {layer}: {str(e)}")
            continue

    return harmonized_data

def harmonize_global(BN_layers, silos,results):

    harmonized_data = {}
    print(f"silo name:{silos}")

    for layer in BN_layers:
        print(f"Harmonizing data for layer: {layer}")
        
        # Extract features from multiple silos
        silo_features = {silo: results[silo][layer] for silo in silos}
        
        for silo, features in silo_features.items():
            print(f"Before reshape Local features {layer} in {silo}: {features.shape}")
            
            # Ensure 2D shape
            silo_features[silo] = features.reshape(-1, 1) if features.ndim == 1 else features
            
            if layer == 'fc3.weight':
                silo_features[silo] = silo_features[silo].T
            
            print(f"After reshape Local features {layer} in {silo}: {silo_features[silo].shape}")
        
        # Concatenate data
        try:
            combined_data = np.vstack([silo_features[silo] for silo in silos])
            print(f"Combined features:{combined_data.shape}")

        except ValueError as e:
            print(f"Dimension mismatch in layer {layer}: {e}")
            continue
        
        # Prepare DataFrame
        feature_columns = [f"feature_{i}" for i in range(combined_data.shape[1])]
        data_df = pd.DataFrame(combined_data, columns=feature_columns)
        batch_labels = [
            silo for silo in silos 
            if silo in silo_features and isinstance(silo_features[silo], np.ndarray) and len(silo_features[silo]) > 0
            for _ in range(silo_features[silo].shape[0])]
        covars = pd.DataFrame({'batch': batch_labels})

        try:
            # Apply NeuroCombat harmonization
            harmonized_results = neuroCombat(
                dat=data_df.T.values,  # NeuroCombat expects (features, samples)
                covars=covars,
                batch_col='batch',
                eb=True,
                parametric=True,
                mean_only=False
            )

            # Extract and validate harmonized data
            if 'data' not in harmonized_results:
                print(f"No harmonized data for layer {layer}.")
                continue
            
            harmonized_values = harmonized_results['data'].T  # Revert to (samples, features)
            harmonized_df = pd.DataFrame(harmonized_values, columns=feature_columns)
            harmonized_df['batch'] = batch_labels

            harmonized_localfeatures = harmonized_df[harmonized_df["batch"] == silos[0]].drop(columns=["batch"]).values
            sum = np.zeros(harmonized_localfeatures.shape)

            for silo in silos:
                harmonized_localfeatures = harmonized_df[harmonized_df["batch"] == silo].drop(columns=["batch"]).values

                if(np.all(sum) == 0):
                    sum = harmonized_localfeatures
                else:
                    sum += harmonized_localfeatures
            sum /= len(silos)
            
            # Transpose for specific layers
            if layer == 'fc3.weight':
                harmonized_data[layer] = sum.T
            else:
                harmonized_data[layer] = sum

            if layer.endswith('.bias'):
                harmonized_data[layer] = sum.squeeze()


            print(f"Harmonized local features shape: {harmonized_data[layer].shape}")
            
            print(f"Successfully harmonized layer: {layer}\n")
        
        except Exception as e:
            print(f"Error harmonizing layer {layer}: {str(e)}\n")
            continue
