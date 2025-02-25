import torch , os, sys
sys.path.append('/home/tanurima/germany/')
import train_silo as sil
import model as m
import config
from brain_age_parcels.utils import eval
import update_wts as wts , avg_wts as avg
import avg_wts as hrm

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
        #break

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