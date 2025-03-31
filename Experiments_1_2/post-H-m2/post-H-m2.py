import os, torch, time as t,numpy as np,sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import matplotlib.pyplot as plt
from data_utils import *
from model import *
from data_utils import preprocess
from training import train_globalmodel,train_localmodelgrads
from evaluation import evaluate_model,predict_age
from harmonize import harmonize_localsilos
from config import INIT_LR, MAX_EPOCHS,DEVICE, MOMENTUM,GRADIENT_CLIP,GLOBAL_SCALER

def post_H_m2(base_dir):
    silodata_path =os.path.join(base_dir,'silo_Datasets')

    all_silos = ['CamCAN', 'SALD','eNki']
    test_files = {silo: os.path.join(silodata_path,f'{silo}/Test_{silo}.csv') for silo in all_silos}
    
    lr = INIT_LR

    f =open(f'results_post-H-m2_l15.txt', 'a')
    dir_path = os.path.join(base_dir,'post-H-m2/train_val_losses') 
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        print(f"Directory created: {dir_path}")

    for epochs in MAX_EPOCHS:
        #losses = {silo : [] for silo in all_silos}
        # Dictionaries to store loss histories for dynamic plotting
        loss_history = {
            'CamCAN': {'train': [], 'val': []},
            'SALD':   {'train': [], 'val': []},
            'eNki':   {'train': [], 'val': []}  # using 'train' to store avg_global_loss from train_globalmodel
        }

        model_global = AgePredictor(input_size=1073).to(DEVICE)
        model_global.initialize_weights(model_global)
        silos = ['CamCAN', 'SALD']
        total_samples = 0

        for silo in silos:
            path = os.path.join(silodata_path,f'{silo}/Train_{silo}.csv')
            X,_,_ = preprocess(path)
            total_samples += X.shape[0]

        model_global = AgePredictor(input_size=1073).to(DEVICE)
        model_global.initialize_weights(model_global)
        global_path = os.path.join(silodata_path,f'eNki/Train_eNki.csv')
        model_global,avg_global_loss,train_losses,val_losses,GLOBAL_SCALER = train_globalmodel(model_global,global_path,total_samples)

        # Log the initial global loss
        loss_history['eNki']['train'].append(train_losses)
        loss_history['eNki']['val'].append(val_losses)
        #zeros_list = [0] * len(loss_history['eNki']['train'])
        loss_history['CamCAN']['train'] = train_losses
        loss_history['CamCAN']['val'] = val_losses
        loss_history['SALD']['train'] = train_losses
        loss_history['SALD']['val'] = val_losses
        
        # Initialize momentum buffers
        velocity = {name: torch.zeros_like(param.data) for name, param in model_global.named_parameters()}
        
        f2 = open(f'train_val_losses/post-H-m2_train_val_loss_localsilo_l15_{epochs}.txt','w')
        silo_scalers = {}

        # Learning rate and scheduler setup
        current_lr = INIT_LR
        st_t = t.perf_counter()

        for epoch in range(epochs):  # Global Epochs

            aggregated_gradients = None
            silo_grads ={}

            for silo in silos: # Local Epochs
                train_path = os.path.join(silodata_path,f'{silo}/Train_{silo}.csv')
                grads, train_losses, val_losses,scaler = train_localmodelgrads(train_path, model_global, total_samples)
                silo_scalers[silo] = scaler
                loss_history[f'{silo}']['train'].extend(train_losses)
                loss_history[f'{silo}']['val'].extend(val_losses)
                f2.write(f'{(epoch+1)}\n')
                f2.write(f'Train on {silo} | epochs: {epoch} | Train MAE error: {sum(train_losses)/len(train_losses)}\n')
                f2.write(f'Validate on {silo} | epochs: {epoch} | Val MAE error: {sum(val_losses)/len(val_losses)}\n')
                f2.write('\n')
                silo_grads[f'{silo}'] = grads
                
            aggregated_gradients = harmonize_localsilos(model_global,silo_grads)


            # Clip gradients
            if GRADIENT_CLIP is not None:
                aggregated_gradients = {
                    k: torch.clamp(v, -GRADIENT_CLIP, GRADIENT_CLIP)
                    for k, v in aggregated_gradients.items()
                }

            # Update global model with momentum
            with torch.no_grad():
                for name, param in model_global.named_parameters():
                    if name.startswith('fc') and name != 'fc3.bias': #print(f"Layer: {name}, Shape: {param.shape}")
                        velocity[name] = MOMENTUM * velocity[name] + aggregated_gradients[name]
                        param -= current_lr * velocity[name]

            print(f"Round {epoch+1}: Updated Global Model Weights")

        f2.close()

        end_t = t.perf_counter()

        f.write(f'Avg val loss on central data:{avg_global_loss}\n')
        f.write(f'Time taken:{(end_t-st_t)/60.0} mins\n')

        for silo, test_path in test_files.items():
            if silo == 'eNki':
                avg_loss = evaluate_model(test_path,model_global,GLOBAL_SCALER)  # Evaluate on each test set
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
        # Flatten nested lists if necessary.
        camcan_train = np.array(loss_history['CamCAN']['train']).flatten()
        camcan_val   = np.array(loss_history['CamCAN']['val']).flatten()
        sald_train   = np.array(loss_history['SALD']['train']).flatten()
        sald_val     = np.array(loss_history['SALD']['val']).flatten()

        # Create full x-axis values based on the length of each list.
        x_camcan = range(1, len(camcan_train) + 1)
        x_camcan_val = range(1, len(camcan_val) + 1)
        x_sald = range(1, len(sald_train) + 1)
        x_sald_val = range(1, len(sald_val) + 1)

        # Create a figure with two subplots side by side.
        fig, (ax_camcan, ax_sald) = plt.subplots(1, 2, figsize=(12, 5))

        # Plot CamCAN losses on the first subplot using all data points.
        ax_camcan.plot(x_camcan, camcan_train, label='Train Loss', color='blue', marker='o', markersize=3)
        ax_camcan.plot(x_camcan_val, camcan_val, label='Validation Loss', color='red', marker='o', markersize=3)
        ax_camcan.set_title("CamCAN Losses")
        ax_camcan.set_xlabel("Epoch")
        ax_camcan.set_ylabel("Loss")
        ax_camcan.legend()
        ax_camcan.grid(True)
        # Set x-ticks with a gap of 50
        ax_camcan.set_xticks(range(1, len(camcan_train) + 1, 50))

        # Plot SALD losses on the second subplot using all data points.
        ax_sald.plot(x_sald, sald_train, label='Train Loss', color='blue', marker='o', markersize=3)
        ax_sald.plot(x_sald_val, sald_val, label='Validation Loss', color='red', marker='o', markersize=3)
        ax_sald.set_title("SALD Losses")
        ax_sald.set_xlabel("Epoch")
        ax_sald.set_ylabel("Loss")
        ax_sald.legend()
        ax_sald.grid(True)
        # Set x-ticks with a gap of 50
        ax_sald.set_xticks(range(1, len(sald_train) + 1, 50))

        # Adjust layout, save the figure, and display it.
        plt.tight_layout()
        plt.savefig(os.path.join(dir_path, f'losses_{epochs}.png'))
        plt.show()
        plt.close(fig)

        del model_global
   
    f.close()
                 
if __name__ =='__main__':
    #base_dir = input("Enter the base directory:") 
    base_dir = "/home/tanurima/germany/brain_age_parcels/Experiments_1_2/"
    post_H_m2(base_dir)