#%%
import os
import torch
import time as t
import numpy as np
import csv
import sys
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

project_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))          # noqa
sys.path.append(project_dir)

from lib.data_utils import preprocess                                                               # noqa
from lib.model import AgePredictor                                                                  # noqa                        
from lib.utils import ensure_dir, plot_loss_curves,set_parameters,select_device                                                  # noqa
from lib.data_utils import preprocess                                                               # noqa
from lib.training import train_globalmodel, train_localmodel                                        # noqa
from lib.evaluation import evaluate_model, predict_age, compute_validation_loss                     # noqa
from lib.harmonize import feature_harmonization                                                     # noqa

#%%
# #####################
# Experiment parameters #######################
Global_epoch = 200 
federated_epochs = 15
dropout_rate = 0.2
current_lr = 1e-3 #8e-4 #1e-3 #5e-4
gradient_clip = 1.0
momentum = 0.8

# Configuration
env_gpu = os.environ.get("CUDA_ID")
DEVICE = select_device(int(env_gpu) if env_gpu is not None else None)

print("Automatically selected:", DEVICE)
set_parameters()  # Set random seed for reproducibility
# #############################################
silodata_path = os.path.join(project_dir, 'Experiments_1_2_eNKI/silo_Datasets')
# TODO change enki
all_silos = ['CamCAN', 'SALD', 'eNKI']
local_silos = ['CamCAN', 'SALD']

test_files = {silo: os.path.join(silodata_path, f'{silo}/Test_{silo}.csv') for silo in all_silos}   # noqa

# Dictionaries to store loss histories for dynamic plotting
loss_history = {
    'CamCAN': {'train': [], 'val': []},
    'SALD':   {'train': [], 'val': []},
    'eNKI':   {'train': [], 'val': []}  # using 'train' to store avg_global_loss from train_globalmodel
}


total_samples = 0
for silo in local_silos:
    path = os.path.join(silodata_path, f'{silo}/Train_{silo}.csv')
    feature_harmonization(silodata_path, silo,'eNKI')  # Harmonize the data for each silo
    X, _, _ = preprocess(path)
    total_samples += X.shape[0]

model_global = AgePredictor(input_size=1073).to(DEVICE)                                                          # noqa
model_global.initialize_weights(model_global)
global_path = os.path.join(silodata_path, f'eNKI/Train_eNKI.csv')
model_global, avg_global_loss, train_losses, val_losses, global_dict = train_globalmodel(model_global,           # noqa
                                                                            global_path,            # noqa
                                                                            DEVICE,FINE_TUNE_EPOCHS=20)          # noqa        

# Save initial global loss to CSV
initial_results_path = os.path.join(project_dir, f'Experiments_1_2_eNKI/post-H-m1/results_post-H-m1/post-H-m1_l{federated_epochs}.csv')
ensure_dir(initial_results_path)  # Ensure the directory exists
#os.makedirs(os.path.dirname(initial_results_path), exist_ok=True)  # Create directory if missing

with open(initial_results_path, 'a', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Description', 'Value'])
    writer.writerow(['Initial global model loss on central data', avg_global_loss])

# Log the initial global loss
loss_history['eNKI']['train'].append(train_losses)
loss_history['eNKI']['val'].append(val_losses)
#zeros_list = [0] * len(loss_history['eNki']['train'])
loss_history['CamCAN']['train'] = train_losses
loss_history['CamCAN']['val'] = val_losses
loss_history['SALD']['train'] = train_losses
loss_history['SALD']['val'] = val_losses

# Initialize momentum buffers
velocity = {name: torch.zeros_like(param.data) for name, param in model_global.named_parameters()}      # noqa

silo_scalers = {}

st_t = t.perf_counter()

best_val_loss_silo = {silo: float('inf') for silo in local_silos}  # Initialize best validation loss for each silo
best_model_silo = {silo: None for silo in local_silos}  # Initialize best model for each silo
# Usage in training loop
best_global_loss = float('inf')
best_model_global = None  # Initialize with no model
global_dict['best_model'] = best_model_global  # Store the best model in the global dict

X_train_silo = {}
y_train_silo = {}
silo_kl, silo_mmd ={},{}

for silo in local_silos:  # Local Epochs
    train_path = os.path.join(silodata_path, f'{silo}/Train_harmonized_{silo}.csv')

    X, y, silo = preprocess(train_path)
    X_train_silo[silo] = X
    y_train_silo[silo] = y
    silo_kl[silo], silo_mmd[silo] = (0.3, 0.5) if silo == 'SALD' else (0.3, 0.5)

csv_loss_path = os.path.join(project_dir, f'Experiments_1_2_eNKI/post-H-m1/results_post-H-m1/train_val_loss_localsilo_l{federated_epochs}.csv')
with open(csv_loss_path, mode='w', newline='') as f2:
    writer = csv.writer(f2)
    writer.writerow(['Epoch', 'Silo', 'Train_MAE', 'Val_MAE'])  # header
    best_global_epoch = 0

    for epoch in range(Global_epoch):
        aggregated_gradients = None
        sum_loss = 0.0

        for silo in local_silos:
            print(f"Round {epoch+1}: Training on {silo}")
            X_silo_current = X_train_silo[silo]
            y_silo_current = y_train_silo[silo]

            grads, train_losses, val_losses, best_val_loss, best_model = train_localmodel(
                X_silo=X_silo_current,
                y_silo=y_silo_current,
                model=model_global,
                total_samples=total_samples,
                epochs=federated_epochs,
                best_val_loss=best_val_loss_silo[silo],
                DEVICE=DEVICE,
                mu_prox=0.05,
                lambda_kl=silo_kl[silo],
                lambda_mmd=silo_mmd[silo])

            best_model_silo[silo] = best_model
            best_val_loss_silo[silo] = best_val_loss
            sum_loss += best_val_loss

            loss_history[f'{silo}']['train'].extend(train_losses)
            loss_history[f'{silo}']['val'].extend(val_losses)

            # Write to CSV
            avg_train_mae = sum(train_losses) / len(train_losses)
            avg_val_mae = sum(val_losses) / len(val_losses)
            writer.writerow([epoch + 1, silo, avg_train_mae, avg_val_mae])

            # Aggregate gradients
            if aggregated_gradients is None:
                aggregated_gradients = grads
            else:
                aggregated_gradients = {
                    key1: g1 + g2
                    for (key1, g1), (key1, g2) in zip(aggregated_gradients.items(), grads.items())
                }

        # Clip gradients
        if gradient_clip is not None:
            aggregated_gradients = {
                k: torch.clamp(v, -gradient_clip, gradient_clip)
                for k, v in aggregated_gradients.items()
            }

        # Update global model with momentum
        with torch.no_grad():
            for name, param in model_global.named_parameters():
                if name.startswith('fc'):
                    velocity[name] = momentum * velocity[name] + aggregated_gradients[name]
                    param -= current_lr * velocity[name]
        
        # After training, validate and update best global model
        best_global_loss, best_model_global, best_global_epoch = compute_validation_loss(
            gmodel=model_global,
            val_loader=global_dict['val_dataloader'],
            criterion=global_dict['criterion'],
            current_best_loss=best_global_loss,
            current_best_model=best_model_global,
            epoch=epoch,
            best_epoch=best_global_epoch,
            device=DEVICE
        )

        print(f"Round {epoch+1}: Updated Global Model Weights")

f2.close()

end_t = t.perf_counter()

global_scaler = StandardScaler()
X_train, _, _ = preprocess(global_path)
global_scaler.fit(X_train)
silo_scalers["eNKI"] = global_scaler  # Store the scaler for eNki

for silo in local_silos:  # Local Epochs
    train_path = os.path.join(silodata_path, f'{silo}/Train_harmonized_{silo}.csv')
    X_train, _, silo = preprocess(train_path)

    silo_scaler = StandardScaler()
    silo_scaler.fit(X_train)
    # Store the scaler for each silo in a dictionary
    silo_scalers[silo] = silo_scaler

results_age_path = os.path.join(project_dir, 'Experiments_1_2_eNKI/post-H-m1/results_post-H-m1/predictions')
# Save final test results to the same CSV
with open(initial_results_path, 'a', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow([])
    writer.writerow(['Final Evaluation Results on best silo(s)'])

    for silo, test_path in test_files.items():
        if silo == 'eNKI':
            scaler_silo = global_scaler
            best_model_sile = best_model_global
        else:
            scaler_silo = silo_scalers[silo]
            best_model_sile = best_model_silo[silo]

        avg_loss = evaluate_model(test_path, best_model_sile, scaler_silo,DEVICE)
        predict_age(best_model_sile, test_path, results_age_path ,global_scaler,'best_local_m')
            
        writer.writerow([f'Test on {silo}', f'Epochs: {Global_epoch}', f'Test MAE: {avg_loss}'])
    writer.writerow([])
    
    writer.writerow([f'Final Evaluation Results on best global model : {best_global_epoch}'])

    for silo, test_path in test_files.items():
        if silo == 'eNKI':
            avg_loss = evaluate_model(test_path, best_model_global, global_scaler,DEVICE)
            predict_age(best_model_global, test_path, results_age_path ,global_scaler,'best_global_m')
        else:
            scaler_silo = silo_scalers[silo]
            avg_loss = evaluate_model(test_path, best_model_global, scaler_silo, DEVICE)
            predict_age(best_model_global, test_path, results_age_path,scaler_silo,'best_global_m')
        writer.writerow([f'Test on {silo}', f'Epochs: {Global_epoch}', f'Test MAE: {avg_loss}'])
    writer.writerow([])
    writer.writerow(['Final Evaluation Results on global model'])

    for silo, test_path in test_files.items():
        if silo == 'eNKI':
            avg_loss = evaluate_model(test_path, model_global, global_scaler,DEVICE)
            predict_age(model_global, test_path, results_age_path ,global_scaler,'global_m')
        else:
            scaler_silo = silo_scalers[silo]
            avg_loss = evaluate_model(test_path, model_global, scaler_silo,DEVICE)
            predict_age(model_global, test_path, results_age_path,scaler_silo,'global_m')
        writer.writerow([f'Test on {silo}', f'Epochs: {Global_epoch}', f'Test MAE: {avg_loss}'])
    writer.writerow([])
    writer.writerow(['Total Time (mins)', (end_t - st_t) / 60.0])

#plot_loss_curves(loss_history, project_dir, 'Experiments_1_2_eNKI/post-H-m1/results_post-H-m1/')  # Plotting function

for silo in local_silos:  # Local Epochs
    train_path = os.path.join(silodata_path, f'{silo}/Train_harmonized_{silo}.csv')
    test_path = os.path.join(silodata_path, f'{silo}/Test_harmonized_{silo}.csv')
    # Remove the file if it exists
    if os.path.exists(train_path) and os.path.exists(test_path):
        os.remove(test_path)
        os.remove(train_path)
        print(f"Deleted: {train_path} and {test_path}")
    else:
        print(f"File not found: {train_path} or {test_path}")
    

del model_global
del best_model_silo
