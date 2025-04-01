import os
import torch
import time as t
import numpy as np
import sys
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

project_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))          # noqa
sys.path.append(project_dir)

from lib.data_utils import preprocess                                                               # noqa
from lib.model import AgePredictor                                                                  # noqa                        
from lib.utils import ensure_dir                                                                    # noqa
from lib.data_utils import preprocess                                                               # noqa
from lib.training import train_globalmodel, train_localmodel                                        # noqa
from lib.evaluation import evaluate_model, predict_age                                              # noqa
from lib.config import DEVICE, GLOBAL_SCALER                               # noqa

# #####################
# Experiment parameters #######################
Global_epoch = 50
federated_epochs = 15
dropout_rate = 0.2
lr = 1e-3
# Learning rate and scheduler setup
current_lr = 1e-3
gradient_clip = 1.0
momentum = 0.8
# #############################################

silodata_path = os.path.join(project_dir, 'silo_Datasets')
# TODO change enki
all_silos = ['CamCAN', 'SALD', 'eNki']
local_silos = ['CamCAN', 'SALD']

test_files = {silo: os.path.join(silodata_path, f'{silo}/Test_{silo}.csv') for silo in all_silos}   # noqa

# TODO: results as .csv
f = open(f'results_pre-H_l15.txt', 'a')

save_path = os.path.join(project_dir, 'pre-H/train_val_losses')

ensure_dir(save_path)

# Dictionaries to store loss histories for dynamic plotting
loss_history = {
    'CamCAN': {'train': [], 'val': []},
    'SALD':   {'train': [], 'val': []},
    'eNki':   {'train': [], 'val': []}  # using 'train' to store avg_global_loss from train_globalmodel
}


total_samples = 0
for silo in local_silos:
    path = os.path.join(silodata_path, f'{silo}/Train_{silo}.csv')
    X, _, _ = preprocess(path)
    total_samples += X.shape[0]

model_global = AgePredictor(input_size=1073, dropout_rate=dropout_rate).to(DEVICE)                                                          # noqa
model_global.initialize_weights(model_global)
global_path = os.path.join(silodata_path, f'eNki/Train_eNki.csv')
model_global, avg_global_loss, train_losses, val_losses = train_globalmodel(model_global,           # noqa
                                                                            global_path,            # noqa
                                                                            total_samples)          # noqa        


# Log the initial global loss
loss_history['eNki']['train'].append(train_losses)
loss_history['eNki']['val'].append(val_losses)
#zeros_list = [0] * len(loss_history['eNki']['train'])
loss_history['CamCAN']['train'] = train_losses
loss_history['CamCAN']['val'] = val_losses
loss_history['SALD']['train'] = train_losses
loss_history['SALD']['val'] = val_losses

f.write(f'Initial global model loss on central data: {avg_global_loss}\n')

# Initialize momentum buffers
velocity = {name: torch.zeros_like(param.data) for name, param in model_global.named_parameters()}      # noqa

f2 = open(f'pre-H_train_val_loss_localsilo_l15_{Global_epoch}.txt', 'w')
silo_scalers = {}

st_t = t.perf_counter()

best_val_loss_silo = {silo: float('inf') for silo in local_silos}  # Initialize best validation loss for each silo
best_model_silo = {silo: None for silo in local_silos}  # Initialize best model for each silo

X_train_silo = {}
y_train_silo = {}

for silo in local_silos:  # Local Epochs
    train_path = os.path.join(silodata_path, f'{silo}/Train_{silo}.csv')

    X, y, silo = preprocess(train_path)
    X_train_silo[silo] = X
    y_train_silo[silo] = y

for epoch in range(Global_epoch):  # Global Epochs

    aggregated_gradients = None

    for silo in local_silos:  # Local Epochs
        # Get the data for the current silo
        X_silo_current = X_train_silo[silo]
        y_silo_current = y_train_silo[silo]
        grads, train_losses, val_losses, best_val_loss, best_model = train_localmodel(X_silo=X_silo_current,
                                                                                      y_silo=y_silo_current,
                                                                                      model_global=model_global,
                                                                                      total_samples=total_samples,
                                                                                      epochs=federated_epochs,
                                                                                      best_val_loss=best_val_loss_silo[silo])
        best_model_silo[silo] = best_model  # Update the best model for the current silo
        best_val_loss_silo[silo] = best_val_loss  # Update the best validation loss for the current silo
        loss_history[f'{silo}']['train'].extend(train_losses)
        loss_history[f'{silo}']['val'].extend(val_losses)
        # TODO to .csv
        f2.write(f'{(epoch+1)}\n')
        f2.write(f'Train on {silo} | epochs: {epoch} | Train MAE error: {sum(train_losses)/len(train_losses)}\n')
        f2.write(f'Validate on {silo} | epochs: {epoch} | Val MAE error: {sum(val_losses)/len(val_losses)}\n')
        f2.write('\n')

        if aggregated_gradients is None:
            aggregated_gradients = grads
        else:
            aggregated_gradients = {key1: g1 + g2 for (key1, g1), (key1, g2) in zip(aggregated_gradients.items(), grads.items())}

    # Clip gradients
    if gradient_clip is not None:
        aggregated_gradients = {
            k: torch.clamp(v, -gradient_clip, gradient_clip)
            for k, v in aggregated_gradients.items()
        }

    # Update global model with momentum
    with torch.no_grad():
        for name, param in model_global.named_parameters():
            velocity[name] = momentum * velocity[name] + aggregated_gradients[name]
            param -= current_lr * velocity[name]

    print(f"Round {epoch+1}: Updated Global Model Weights")

f2.close()

end_t = t.perf_counter()

f.write(f'Avg val loss on central data:{avg_global_loss}\n')
f.write(f'Time taken:{(end_t-st_t)/60.0} mins\n')

global_scaler = StandardScaler()
X_train, _, _ = preprocess(global_path)
global_scaler.fit(X_train)
silo_scalers["eNki"] = global_scaler  # Store the scaler for eNki

for silo in local_silos:  # Local Epochs
    train_path = os.path.join(silodata_path, f'{silo}/Train_{silo}.csv')
    X_train, _, silo = preprocess(train_path)

    silo_scaler = StandardScaler()
    silo_scaler.fit(X_train)
    # Store the scaler for each silo in a dictionary
    silo_scalers[silo] = silo_scaler

# TODO: check Global_epoch
for silo, test_path in test_files.items():
    if silo == 'eNki':
        avg_loss = evaluate_model(test_path, model_global, global_scaler)  # Evaluate on each test set
        predict_age(model_global, test_path, global_scaler, Global_epoch)
        print(f'Test on {silo} | epochs: {Global_epoch} | Test MAE error: {avg_loss}')
        f.write(f'Test on {silo} | epochs: {Global_epoch} | Test MAE error: {avg_loss}\n')
    else:
        scaler_silo = silo_scalers[silo]
        best_model_sile = best_model_silo[silo]

        avg_loss = evaluate_model(test_path, best_model_sile, scaler_silo)
        predict_age(best_model_sile, test_path, scaler_silo, Global_epoch)
        print(f'Test on {silo} | epochs: {Global_epoch} | Test MAE error: {avg_loss}')
        f.write(f'Test on {silo} | epochs: {Global_epoch} | Test MAE error: {avg_loss}\n')
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
