import torch , os, sys
import torch.nn as nn
sys.path.append('/home/tanurima/germany/')
import train_silo as sil
import model as m
import config
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from brain_age_parcels.utils import *
import update_wts as wts , avg_wts as avg
import avg_wts as hrm

def evaluate_model(test_path, model_path):

    X_test,y_test,silo_name = utils.preprocess(test_path)
    # Load the model (assuming model_name corresponds to a model function or class)

    model = m.AgePredictor(X_test.shape[1])
    model.load_state_dict(torch.load(model_path, map_location=config.DEVICE))
    model.to(config.DEVICE)
    model.eval()

    scaler = StandardScaler()
    X_test = scaler.fit_transform(X_test)

    # Convert to tensors
    test_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_test), torch.FloatTensor(y_test)),
        batch_size=config.BATCH_SIZE, shuffle=True
    )
    
    total_loss = 0.0
    num_batches = 0
    criterion = nn.L1Loss()
    
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(config.DEVICE), y_batch.to(config.DEVICE)
            y_batch = y_batch.view(-1, 1)
            predictions = model(X_batch).view(-1, 1)
            loss = criterion(predictions, y_batch)
            total_loss += loss.item()
            num_batches += 1

    avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')
    print(f'\nTest error of {silo_name} : {avg_loss}\n')
    return avg_loss

def FL_weightedlocalgrads():

    silos = {0: 'CamCAN', 1: 'SALD'}
    model = m.AgePredictor(input_size=1073).to(config.DEVICE)
    model.initialize_weights(model)
    torch.save(model.state_dict(), 'globalModel_0.pt')

    layers = []
    for name, param in model.state_dict().items():
        #print(f"Parameter: {name}, shape: {param.shape}")
        layers.append(name)

    silo_loss = {name: [] for name in silos.values()}

    for epoch in range(config.NUM_EPOCHS):

        print(f"\n=== Global Epoch {epoch+1}/{config.NUM_EPOCHS} ===")
        global_path = f'globalModel_{epoch}.pt'
        print(f"Initially the path of global model: {global_path}")
        
        results = {}
        
        for i in list(silos.keys()):
            silo = silos[i]
            print(f'silo name:{silo}')
            if silo == 'SALD': max_epochs = 5
            else : max_epochs = 10
            results[silo] = sil.localsilo(silo, global_path, max_epochs)
            
            #for layer_name, grad in results[silo].items():
                #if grad is not None and layer_name not in layers:
                #    layers.append(layer_name)

        print(f"Local gradients averaging globally and the layers: {layers}\n")
        avg_results = avg.avg_localsilo(layers, list(silos.values()), results)

        #for key, value in avg_results.items():
        #    print(f'layer name:{key} and gradient value :{value}')
        #    break
        
        model = wts.update_model_weights(avg_results, global_path)
        model_path = f'globalModel_{epoch+1}.pt'
        torch.save(model.state_dict(), model_path)
        print(f"{epoch+1}, global model weights updated")

        for i in list(silos.keys()):
            silo = silos[i]
            test_path = f'/home/tanurima/germany/brain_age_parcels/{silo}/{silo}_test.csv'
            silo_loss[silo].append(evaluate_model(test_path, model_path))
        
        os.remove(f'globalModel_{epoch}.pt')
        print(f"model Deleted: globalModel_{epoch}.pt")
    
    return epoch+1, model_path, silo_loss


def FL_weightedlocalglobalgrads():

    silos = {0: 'CamCAN', 1: 'SALD'}
    model = m.AgePredictor(input_size=1073).to(config.DEVICE)
    model.initialize_weights(model)
    torch.save(model.state_dict(), 'globalModel_0.pt')

    layers = []
    for name, param in model.state_dict().items():
        #print(f"Parameter: {name}, shape: {param.shape}")
        layers.append(name)

    silo_loss = {name: [] for name in silos.values()}
    
    for epoch in range(config.NUM_EPOCHS):

        print(f"\n=== Global Epoch {epoch+1}/{config.NUM_EPOCHS} ===")
        global_path = f'globalModel_{epoch}.pt'
        print(f"Initially the path of global model: {global_path}")
        
        results,local_grads = {},{}
        
        for i in list(silos.keys()):
            silo = silos[i]
            print(f'silo name:{silo}')
            if silo == 'SALD': max_epochs = 5
            else : max_epochs = 10
            results[silo] = sil.localsilo(silo, global_path, max_epochs)

            local_grads[silo] = avg.avg_localglobal(layers, silo, results[silo])

        print(f"Local gradients averaging globally and the layers: {layers}\n")
        avg_results = avg.avg_localglobal(layers, list(silos.values()), local_grads)

        #for key, value in avg_results.items():
        #    print(f'layer name:{key} and gradient value :{value}')
        #    break
        
        model = wts.update_model_weights(avg_results, global_path)
        model_path = f'globalModel_{epoch+1}.pt'
        torch.save(model.state_dict(), model_path)
        print(f"{epoch+1}, global model weights updated")

        for i in list(silos.keys()):
            silo = silos[i]
            test_path = f'/home/tanurima/germany/brain_age_parcels/{silo}/{silo}_test.csv'
            silo_loss[silo].append(evaluate_model(test_path, model_path))

        os.remove(f'globalModel_{epoch}.pt')
        print(f"model Deleted: globalModel_{epoch}.pt")

    return epoch+1, model_path, silo_loss

#silos = ['CamCAN', 'SALD']
#FL_weightedlocalgrads()





"""
for i in list(silos.keys()):
    silo = silos[i]
    test_path = f'/home/tanurima/germany/brain_age_parcels/{silo}/{silo}_features_cleaned_test_data.csv'
    silo_loss[silo].append(eval.evaluate_model(test_path, model_path))
"""