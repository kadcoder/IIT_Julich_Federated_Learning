import os, torch, time as t,numpy as np
from utils import *
from model import *
from data_utils import preprocess
from training import train_globalmodel,train_localmodel
from evaluation import evaluate_model,predict_age
from config import INIT_LR, MAX_EPOCHS,DEVICE, MOMENTUM,GRADIENT_CLIP,GLOBAL_SCALER

def pre_H(base_dir):

    silodata_path = os.path.join(base_dir,'Experiments_1_2/silo_Datasets')
    all_silos = ['CamCAN', 'SALD','eNki']
    test_files = {silo: os.path.join(silodata_path,f'{silo}/Test_{silo}.csv') for silo in all_silos}
    
    lr = INIT_LR

    f = open(f'results_pre-H_l15.txt', 'w')
    dir_path = os.path.join(base_dir,'pre-H/train_val_losses')
	
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        print(f"Directory created: {dir_path}")
    else:
        print(f"Directory has been created: {dir_path}")

    for epochs in MAX_EPOCHS:
        losses = {silo : [] for silo in all_silos}

        silos = ['CamCAN', 'SALD']
        total_samples = 0

        for silo in silos:
            path = os.path.join(silodata_path,f'{silo}/Train_{silo}.csv')
            X,_,_ = preprocess(path)
            total_samples += X.shape[0]

        model_global = AgePredictor(input_size=1073).to(DEVICE)
        model_global.initialize_weights(model_global)
        global_path = os.path.join(silodata_path,f'eNki/Train_eNki.csv')
        model_global,avg_global_loss = train_globalmodel(model_global,global_path,total_samples)

        # Initialize momentum buffers
        velocity = {name: torch.zeros_like(param.data) for name, param in model_global.named_parameters()}
        
        f2 = open(f'train_val_losses/pre-H_train_val_loss_localsilo_l15_{epochs}.txt','w')
        silo_scalers ={}

        # Learning rate and scheduler setup
        current_lr = INIT_LR
        
        st_t = t.perf_counter()
        for epoch in range(epochs):  # Global Epochs

            aggregated_gradients = None
            
            for silo in silos: # Local Epochs
                train_path = os.path.join(silodata_path,f'{silo}/Train_{silo}.csv')
                grads, train_loss, val_loss,scaler = train_localmodel(train_path, model_global, total_samples)
                silo_scalers[silo] = scaler
                f2.write(f'{(epoch+1)}\n')
                f2.write(f'Train on {silo} | epochs: {epoch} | Train MAE error: {train_loss}\n')
                f2.write(f'Validate on {silo} | epochs: {epoch} | Val MAE error: {val_loss}\n')
                f2.write('\n')
                
                
                if aggregated_gradients is None:
                    aggregated_gradients = grads
                else: 
                    aggregated_gradients = {key1 : g1 + g2 for (key1, g1), (key1, g2) in zip(aggregated_gradients.items(), grads.items())}


            # Clip gradients
            if GRADIENT_CLIP is not None:
                aggregated_gradients = {
                    k: torch.clamp(v, -GRADIENT_CLIP, GRADIENT_CLIP)
                    for k, v in aggregated_gradients.items()
                }

            # Update global model with momentum
            with torch.no_grad():
                for name, param in model_global.named_parameters():
                    velocity[name] = MOMENTUM * velocity[name] + aggregated_gradients[name]
                    param -= current_lr * velocity[name]

            print(f"Round {epoch+1}: Updated Global Model Weights")

        f2.close()

        end_t = t.perf_counter()

        f.write(f'Avg val loss on central data:{avg_global_loss}\n')
        f.write(f'Time taken:{(end_t-st_t)/60.0} mins\n')

        for silo, test_path in test_files.items():
            if silo == 'eNki':
                avg_loss = evaluate_model(test_path,model_global)  # Evaluate on each test set
                predict_age(model_global,test_path,GLOBAL_SCALER,epochs)
                print(f'Test on {silo} | epochs: {epochs} | Test MAE error: {avg_loss}')
                f.write(f'Test on {silo} | epochs: {epochs} | Test MAE error: {avg_loss}\n')
            else:
                scaler = silo_scalers[silo]
                avg_loss = evaluate_model(test_path,model_global,scaler)
                predict_age(model_global,test_path,scaler,epochs)
                print(f'Test on {silo} | epochs: {epochs} | Test MAE error: {avg_loss}')
                f.write(f'Test on {silo} | epochs: {epochs} | Test MAE error: {avg_loss}\n')
        f.write('\n')

        del model_global
       
    f.close()

if __name__=='__main__':
    base_dir = input("Enter the base directory:")
    pre_H(base_dir)