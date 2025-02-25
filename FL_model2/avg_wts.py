import pandas as pd
import numpy as np

def avg_localsilo(layers, silos, grads):

    averaged_data = {}
    BN_layers = [layer for layer in layers if not layer.startswith('bn') and not layer.endswith('bias')] #and not layer =='fc3.weight'


    for layer in BN_layers:
        print(f"Harmonizing data for layer: {layer}")

        # Validate layer presence
        #if layer not in silo1_grads or layer not in silo2_grads:
        #    raise KeyError(f"Layer {layer} missing in local or global gradients.")

        avg_grads = None 
        # Extract features
        for silo in silos:
            print('local features : ',type(grads[silo][f'{silo}_local'][layer]))
            if avg_grads is None:
                avg_grads = np.array(grads[silo][f'{silo}_local'][layer])
            else:
                avg_grads += np.array(grads[silo][f'{silo}_local'][layer])
            
        avg_grads = avg_grads / len(silos)

        averaged_data[layer] = avg_grads

    return averaged_data


def avg_localglobal(layers, silo ,silo_grads):

    averaged_data = {}
    BN_layers = [layer for layer in layers if not layer.startswith('bn') and not layer.endswith('bias')] #and not layer =='fc3.weight'


    for layer in BN_layers:
        print(f"Harmonizing data for layer: {layer}")

        # Validate layer presence
        if layer not in silo1_grads or layer not in silo2_grads:
            raise KeyError(f"Layer {layer} missing in local or global gradients.")
        
       
        local_features = np.array(silo_grads[f'{silo}_local'][layer])
        global_features = np.array(silo_grads[f'{silo}_global'][layer])

        print(f"Before reshape Local features of silo1 {layer}: {silo1_features.shape}, local features of silo2 {layer}: {silo2_features.shape}")

        avg_grads = (local_features + global_features) / 2

        averaged_data[layer] = avg_grads

    return averaged_data
