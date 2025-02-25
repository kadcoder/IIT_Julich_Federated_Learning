import train_localsilo as silo
import train_global as gl
import update_wts as wts

def localsilo(silo_name,global_path,epochs,global_name ='eNki'):

    results = {}
    
    # Load and preprocess data
    local_path = f'/home/tanurima/germany/brain_age_parcels/{silo_name}/{silo_name}_train.csv'
    test_path = f'/home/tanurima/germany/brain_age_parcels/{global_name}/{global_name}_train.csv'
    
    
    local_gradients,scaler = silo.train_localmodel(local_path, global_path,epochs)
    results[f'{silo_name}_local'] = local_gradients
    global_gradients = gl.train_globalmodel(test_path, silo_name)
    results[f'{silo_name}_global'] = global_gradients

    #layers = list(local_gradients.keys())
    #print(f'layers in the each silo {silo_name} :{layers}')

    return results
